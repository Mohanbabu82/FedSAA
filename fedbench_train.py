#!/usr/bin/env python3
"""Stage 2: train every strategy on the cached features (pure NumPy, CPU-friendly, resumable).

Usage:
    python fedbench_train.py --feats feats --out results --stage tune --budget-min 40
    python fedbench_train.py --feats feats --out results --stage main --budget-min 40
Run each stage repeatedly until it prints "stage complete"; finished runs are skipped.

Model on the cached 256-d feature x of client i:
    linear adapter     h = x + s B A x            (s = alpha / r)
    nonlinear adapter  h = x + s B relu(A x)
    no adapter         h = x                      (linear probe)
    logits = W_i h + c_i, with a private classifier per client (classes of other specialties masked).
All clients of a run share one initialization: A ~ N(0, 1/d), B = 0, W ~ N(0, 0.01^2), c = 0.
Each round is E local epochs with Adam; the optimizer state is reset at the start of every round for
every strategy. All 15 clients take part in every round.

Strategies (server step after each round; classifiers are never aggregated):
    local       no communication
    centralized one shared adapter and one classifier per specialty, trained on the pooled data;
                each step averages the gradients of one mini-batch from every client
    fedavg      sample-weighted average of A and of B
    fedprox     fedavg + proximal term mu/2 ||theta - theta_global||^2 on the adapter
    oracle      fedavg restricted to the clients of the same specialty (specialty-oracle clustered FL)
    flexlora    sample-weighted average of the products BA, rank-r truncated SVD, same factors for all
    ffa         FFA-LoRA: A frozen at its (shared) initialization, B trained and averaged
    fedsa       FedSA-LoRA: A averaged, B kept local
    fedsaa      FedSAA, retention in factor space (rho = 0.3)
    fedsaa_r0   FedSAA without retention (rho = 0)
    fedsaa_u    FedSAA-U, retention in update space: (1-rho) U~_i + rho U_i, then one truncated SVD
    probe       local, no adapter (linear probe)
Product-space strategies (flexlora, fedsaa*) are defined for the linear adapter only.
"""
import argparse
import json
import math
import os
import time

import numpy as np

# ------------------------------------------------------------------ configuration
RANK, ALPHA = 16, 32.0
LOCAL_EPOCHS, BATCH = 3, 64
ROUNDS, CHECKPOINTS = 200, (40, 100, 200)
LOG_ROUNDS = (1, 10, 40, 100, 200)
TAU, LAM, RHO = 0.5, 0.5, 0.3
LR_GRID = (3e-4, 1e-3, 3e-3)
MU_GRID = (1e-3, 1e-2, 1e-1)
TUNE_SEED = 1000
TUNE_ROUNDS, TUNE_CHECKPOINTS = 100, (40, 100)
MAIN_SEEDS = tuple(range(10))
if os.environ.get("FEDBENCH_QUICK"):            # smoke test only
    ROUNDS, CHECKPOINTS, LOG_ROUNDS = 6, (2, 4, 6), (1, 6)
    TUNE_ROUNDS, TUNE_CHECKPOINTS, MAIN_SEEDS = 4, (2, 4), tuple(range(3))

STRATS = {
    "linear": ["local", "centralized", "fedavg", "fedprox", "oracle", "flexlora", "ffa", "fedsa",
               "fedsaa", "fedsaa_r0", "fedsaa_u"],
    "nonlinear": ["local", "centralized", "fedavg", "fedprox", "oracle", "ffa", "fedsa"],
    "none": ["probe"],
}
PRODUCT = {"flexlora", "fedsaa", "fedsaa_r0", "fedsaa_u"}


# ------------------------------------------------------------------ data
class Split:
    def __init__(self, Xs, ys):
        self.C = len(Xs)
        self.n = np.array([len(y) for y in ys])
        N, d = self.n.max(), Xs[0].shape[1]
        self.X = np.zeros((self.C, N, d), np.float32)
        self.y = np.zeros((self.C, N), np.int64)
        self.valid = np.zeros((self.C, N), bool)
        for c, (X, y) in enumerate(zip(Xs, ys)):
            self.X[c, :len(y)], self.y[c, :len(y)], self.valid[c, :len(y)] = X, y, True


def load_preset(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    C = meta["n_clients"]
    sp = {k: Split([z[f"X{k}_{c}"] for c in range(C)], [z[f"y{k}_{c}"] for c in range(C)])
          for k in ("tr", "va", "te")}
    return meta, sp


# ------------------------------------------------------------------ model
def init_params(rngs, C, d, K, adapter):
    """One initialization per replica (its own seed), shared by all clients of that replica."""
    W = np.stack([r.normal(0, 0.01, (K, d)) for r in rngs]).astype(np.float32)
    P = {"W": np.repeat(W[:, None], C, 1), "c": np.zeros((len(rngs), C, K), np.float32)}
    if adapter != "none":
        A = np.stack([r.normal(0, 1 / math.sqrt(d), (RANK, d)) for r in rngs]).astype(np.float32)
        P["A"] = np.repeat(A[:, None], C, 1)
        P["B"] = np.zeros((len(rngs), C, d, RANK), np.float32)
    return P


def T(x):
    return np.swapaxes(x, -1, -2)


def forward(P, X, kmask, adapter):
    """X: (R, C, b, d) or (C, b, d), broadcast over replicas.

    logits = W h + c with h = x + s B u, computed as  x W^T + s u (W B)^T  so that the (b, d) array h is
    never formed. Returns the logits and a cache for the backward pass."""
    s = np.float32(ALPHA / RANK)
    logits = np.matmul(X, T(P["W"]))
    z = u = WB = None
    if adapter != "none":
        z = np.matmul(X, T(P["A"]))                                     # (R, C, b, r)
        u = z if adapter == "linear" else np.maximum(z, 0)
        WB = np.matmul(P["W"], P["B"])                                  # (R, C, K, r)
        logits += s * np.matmul(u, T(WB))
    logits += P["c"][..., None, :]
    logits = np.where(kmask[None, :, None, :], logits, np.float32(-1e9))
    return logits, (X, z, u, WB)


def softmax(lg):
    lg = lg - lg.max(-1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(-1, keepdims=True)


def loss_and_grads(P, X, y, wrow, kmask, adapter):
    """Mean cross-entropy per (replica, client), rows weighted by wrow, and its gradients."""
    logits, (X, z, u, WB) = forward(P, X, kmask, adapter)
    sh = logits - logits.max(-1, keepdims=True)
    lse = np.log(np.exp(sh).sum(-1))
    loss = -((np.take_along_axis(sh, y[..., None], -1)[..., 0] - lse) * wrow).sum(-1)
    p = softmax(logits)
    py = np.take_along_axis(p, y[..., None], -1)[..., 0]
    g = p
    np.put_along_axis(g, y[..., None], py[..., None] - 1.0, -1)
    g *= wrow[..., None]
    gT = T(g)                                                            # (R, C, K, b)
    G = {"W": np.matmul(gT, X), "c": g.sum(-2)}
    if adapter != "none":
        s = np.float32(ALPHA / RANK)
        gu = np.matmul(gT, u)                                            # (R, C, K, r)
        G["W"] += s * np.matmul(gu, T(P["B"]))                           # g^T (s u B^T)
        G["B"] = s * np.matmul(T(P["W"]), gu)                            # s W^T g^T u
        Gu = s * np.matmul(g, WB)                                        # (R, C, b, r)
        Gz = Gu if adapter == "linear" else Gu * (z > 0)
        G["A"] = np.matmul(T(Gz), X)                                     # (R, C, r, d)
    return loss, G


class Adam:
    """Adam over (replica, client)-stacked parameters; rows whose mask is False are left untouched."""

    def __init__(self, P, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = np.asarray(lr, np.float32).reshape(-1, 1)            # one learning rate per replica
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.t = {k: np.zeros(v.shape[:2], np.float32) for k, v in P.items()}

    def step(self, P, G, masks):
        for k, g in G.items():
            mk = masks[k]
            if not mk.any():
                continue
            ex = (1,) * (g.ndim - 2)
            full = bool(mk.all())
            self.t[k] += mk
            m, v = self.m[k], self.v[k]
            if full:
                m *= self.b1; m += (1 - self.b1) * g
                g *= g
                v *= self.b2; v += (1 - self.b2) * g
            else:
                mm = mk.reshape(mk.shape + ex).astype(np.float32)
                m += mm * ((1 - self.b1) * (g - m))
                v += mm * ((1 - self.b2) * (g * g - v))
            t = np.maximum(self.t[k], 1).reshape(mk.shape + ex)
            step = (self.lr.reshape(self.lr.shape + ex) / (1 - self.b1 ** t)).astype(np.float32)
            den = np.sqrt(v * (1 / (1 - self.b2 ** t)).astype(np.float32))
            den += self.eps
            upd = np.divide(m, den)
            upd *= step
            if not full:
                upd *= mk.reshape(mk.shape + ex)
            P[k] -= upd


# ------------------------------------------------------------------ server-side aggregation
def factors_from(U):
    """Rank-r truncated SVD of each matrix in U (C, d, d); returns A', B' and retained energy."""
    Pm, S, Qt = np.linalg.svd(U.astype(np.float32))
    sq = np.sqrt(S[:, :RANK])
    B = Pm[:, :, :RANK] * sq[:, None, :]
    A = sq[:, :, None] * Qt[:, :RANK, :]
    energy = (S[:, :RANK] ** 2).sum(1) / np.maximum((S ** 2).sum(1), 1e-30)
    return A.astype(np.float32), B.astype(np.float32), energy


def group_mean(x, w, groups):
    """x: (R, C, ...); weighted mean over the clients of each group, per replica."""
    out = np.empty_like(x)
    for g in np.unique(groups):
        idx = groups == g
        ww = (w[idx] / w[idx].sum()).astype(np.float32)
        out[:, idx] = np.tensordot(x[:, idx], ww, axes=([1], [0]))[:, None]
    return out


def aggregate(strategy, P, n, spec, logs):
    R, C = P["A"].shape[:2]
    allg = np.zeros(C, int)
    w = n.astype(np.float64)
    if strategy in ("fedavg", "fedprox"):
        P["A"], P["B"] = group_mean(P["A"], w, allg), group_mean(P["B"], w, allg)
    elif strategy == "oracle":
        P["A"], P["B"] = group_mean(P["A"], w, spec), group_mean(P["B"], w, spec)
    elif strategy == "ffa":
        P["B"] = group_mean(P["B"], w, allg)
    elif strategy == "fedsa":
        P["A"] = group_mean(P["A"], w, allg)
    elif strategy == "flexlora":
        for r in range(R):
            U = np.matmul(P["B"][r], P["A"][r]).astype(np.float64)
            G = np.tensordot(w / w.sum(), U, axes=1)
            A, B, e = factors_from(G[None])
            P["A"][r], P["B"][r] = A[0], B[0]
            logs[r]["energy"] = np.repeat(e, C)
    elif strategy.startswith("fedsaa"):
        rho = 0.0 if strategy == "fedsaa_r0" else RHO
        for r in range(R):
            U = np.matmul(P["B"][r], P["A"][r])                             # (C, d, d)
            flat = U.reshape(C, -1)
            nrm = np.linalg.norm(flat, axis=1)
            S = (flat @ flat.T) / np.maximum(np.outer(nrm, nrm), 1e-12)
            lg = S / TAU
            Wt = np.exp(lg - lg.max(1, keepdims=True))
            Wt /= Wt.sum(1, keepdims=True)
            Ut = (1 - LAM) * np.tensordot(Wt, U, axes=1) + LAM * U.mean(0, keepdims=True)
            if strategy == "fedsaa_u":
                Ut = (1 - rho) * Ut + rho * U
            A, B, e = factors_from(Ut)
            if strategy == "fedsaa":
                A = (1 - rho) * A + rho * P["A"][r]
                B = (1 - rho) * B + rho * P["B"][r]
            P["A"][r], P["B"][r] = A, B
            logs[r].update(S=S, Wt=Wt, energy=e)


# ------------------------------------------------------------------ evaluation
def predict_proba(P, split, kmask, adapter):
    logits, _ = forward(P, split.X, kmask, adapter)
    return softmax(logits)


def macro_f1(y, yp, labels):
    f = []
    for k in labels:
        tp = np.sum((yp == k) & (y == k)); fp = np.sum((yp == k) & (y != k)); fn = np.sum((yp != k) & (y == k))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def auc_ovr(y, p, K):
    from scipy.stats import rankdata
    out = []
    for k in range(K):
        pos = y == k
        npos, nneg = pos.sum(), (~pos).sum()
        if npos == 0 or nneg == 0:
            continue
        r = rankdata(p[:, k])
        out.append((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return float(np.mean(out)) if out else float("nan")


def full_metrics(prob, split, nclass):
    """prob: (C, N, K) for one replica."""
    acc, f1, auc, correct = [], [], [], []
    for c in range(split.C):
        n = split.n[c]
        y, pr = split.y[c, :n], prob[c, :n, :nclass[c]]
        yp = pr.argmax(1)
        ok = yp == y
        correct.append(ok)
        acc.append(float(ok.mean()))
        f1.append(macro_f1(y, yp, np.union1d(np.unique(y), np.unique(yp))))
        auc.append(auc_ovr(y, pr, nclass[c]))
    acc = np.array(acc)
    return {"acc_eq": float(acc.mean()), "acc_w": float(np.concatenate(correct).mean()),
            "f1": float(np.nanmean(f1)), "auc": float(np.nanmean(auc)), "worst": float(acc.min()),
            "per_client": acc.tolist(), "f1_client": f1, "auc_client": auc}, correct


def accs(prob, split):
    """prob: (R, C, N, K) -> equal-weight accuracy per replica."""
    ok = (prob.argmax(-1) == split.y[None]) & split.valid[None]
    return (ok.sum(-1) / split.n[None]).mean(-1)


# ------------------------------------------------------------------ one run
def run_batch(meta, sp, adapter, strategy, replicas, rounds=ROUNDS, checkpoints=CHECKPOINTS, keep=True):
    """Train one strategy for several replicas at once; replicas = [(seed, lr, mu), ...]."""
    tr, va, te = sp["tr"], sp["va"], sp["te"]
    C, d = tr.C, tr.X.shape[2]
    R = len(replicas)
    nclass = np.array(meta["n_classes"])
    K = int(nclass.max())
    kmask = np.arange(K)[None, :] < nclass[:, None]
    spec = np.array(meta["client_specialty"])
    rngs = [np.random.default_rng(seed) for seed, _, _ in replicas]
    lr = np.array([x[1] for x in replicas], np.float32)
    mu = np.array([x[2] for x in replicas], np.float32)
    P = init_params(rngs, C, d, K, adapter)
    trainable = {k: np.ones((R, C), bool) for k in P}
    if strategy == "ffa":
        trainable["A"][:] = False
    tied = strategy == "centralized"
    steps = int(math.ceil(tr.n.max() / BATCH))
    cidx = np.arange(C)[None, :, None]
    outs = [{"test_curve": [], "val_curve": [], "energy_curve": [], "checkpoints": {}, "logs": {}} for _ in range(R)]
    corrects = [{} for _ in range(R)]
    for t in range(1, rounds + 1):
        glob = {k: P[k].copy() for k in ("A", "B")} if strategy == "fedprox" else None
        opt = Adam(P, lr)
        for _ in range(LOCAL_EPOCHS):
            perm = np.zeros((R, C, steps * BATCH), np.int64)
            for r in range(R):
                for c in range(C):
                    perm[r, c, :tr.n[c]] = rngs[r].permutation(tr.n[c])
            for s in range(steps):
                lo = s * BATCH
                cnt = np.clip(tr.n - lo, 0, BATCH)                       # (C,)
                active_c = cnt > 0
                if not active_c.any():
                    break
                idx = perm[:, :, lo:lo + BATCH]
                val = (np.arange(BATCH)[None, :] < cnt[:, None])          # (C, b)
                Xb = tr.X[cidx, idx]                                      # (R, C, b, d)
                yb = tr.y[cidx, idx]
                wrow = np.broadcast_to((val / np.maximum(cnt, 1)[:, None]).astype(np.float32), (R, C, BATCH))
                _, G = loss_and_grads(P, Xb, yb, wrow, kmask, adapter)
                active = np.broadcast_to(active_c, (R, C))
                if glob is not None:
                    for k in ("A", "B"):
                        G[k] += mu.reshape(R, 1, 1, 1) * (P[k] - glob[k])
                if tied:
                    masks = {}
                    am = active_c.astype(np.float32)
                    for k in list(G):
                        ex = (1,) * (G[k].ndim - 2)
                        if k in ("A", "B"):
                            G[k] = np.repeat((G[k] * am.reshape((1, C) + ex)).sum(1, keepdims=True) / am.sum(), C, 1)
                            masks[k] = np.ones((R, C), bool)
                        else:
                            Gk = np.zeros_like(G[k])
                            mk = np.zeros((R, C), bool)
                            for g in np.unique(spec):
                                gi = spec == g
                                ga = gi & active_c
                                if ga.any():
                                    Gk[:, gi] = G[k][:, ga].mean(1, keepdims=True)
                                    mk[:, gi] = True
                            G[k], masks[k] = Gk, mk
                else:
                    masks = {k: active for k in G}
                masks = {k: masks[k] & trainable[k] for k in masks}
                opt.step(P, G, masks)
        logs = [{} for _ in range(R)]
        if adapter != "none" and strategy not in ("local", "centralized", "probe"):
            aggregate(strategy, P, tr.n, spec, logs)
        pte = predict_proba(P, te, kmask, adapter)
        a_te, a_va = accs(pte, te), accs(predict_proba(P, va, kmask, adapter), va)
        if not np.all(np.isfinite(a_te)):
            raise FloatingPointError("non-finite accuracy")
        for r in range(R):
            o = outs[r]
            o["test_curve"].append(float(a_te[r]))
            o["val_curve"].append(float(a_va[r]))
            if "energy" in logs[r]:
                o["energy_curve"].append(float(np.mean(logs[r]["energy"])))
            if t in LOG_ROUNDS and logs[r]:
                o["logs"][t] = {k: np.asarray(v).round(5).tolist() for k, v in logs[r].items()}
            if t in checkpoints:
                m, corr = full_metrics(pte[r], te, nclass)
                m["val_acc_eq"] = float(a_va[r])
                o["checkpoints"][t] = m
                if keep:
                    corrects[r][t] = corr
    return outs, corrects


# ------------------------------------------------------------------ orchestration
def key_of(stage, preset, adapter, strategy, seed, lr, mu):
    return f"{stage}|{preset}|{adapter}|{strategy}|{seed}|{lr:g}|{mu:g}"


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {json.loads(l)["key"] for l in fh if l.strip()}


def tune_jobs(presets, adapters):
    jobs = []
    for preset in presets:
        for adapter, strats in STRATS.items():
            if adapter not in adapters:
                continue
            for st in strats:
                for lr in LR_GRID:
                    for mu in (MU_GRID if st == "fedprox" else (0.0,)):
                        jobs.append((preset, adapter, st, TUNE_SEED, lr, mu))
    return jobs


def select_configs(path):
    """Best lr (and mu) per preset x adapter x strategy: mean validation accuracy at the checkpoints."""
    best = {}
    with open(path) as fh:
        for l in fh:
            r = json.loads(l)
            if r["stage"] != "tune":
                continue
            score = float(np.mean([r["checkpoints"][str(t)]["val_acc_eq"] for t in TUNE_CHECKPOINTS]))
            k = (r["preset"], r["adapter"], r["strategy"])
            if k not in best or score > best[k][0]:
                best[k] = (score, r["lr"], r["mu"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="feats")
    ap.add_argument("--out", default="results")
    ap.add_argument("--stage", choices=["tune", "main"], required=True)
    ap.add_argument("--budget-min", type=float, default=40.0)
    ap.add_argument("--presets", default="A,B")
    ap.add_argument("--seeds", type=int, default=len(MAIN_SEEDS))
    ap.add_argument("--adapters", default="linear,nonlinear,none")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    jl = os.path.join(args.out, "runs.jsonl")
    presets = args.presets.split(",")
    adapters = args.adapters.split(",")
    data = {p: load_preset(os.path.join(args.feats, f"preset_{p}.npz")) for p in presets}
    for p, (meta, _) in data.items():                 # keep the client metadata with the results
        with open(os.path.join(args.out, f"meta_{p}.json"), "w") as fh:
            json.dump(meta, fh)

    if args.stage == "tune":
        jobs = [("tune",) + j for j in tune_jobs(presets, adapters)]
    else:
        tj = [j for j in tune_jobs(presets, adapters)]
        have = done_keys(jl)
        missing = [j for j in tj if key_of("tune", *j) not in have]
        if missing:
            raise SystemExit(f"tuning incomplete: {len(missing)} tuning runs missing; run --stage tune first")
        best = select_configs(jl)
        with open(os.path.join(args.out, "selected_configs.json"), "w") as fh:
            json.dump({"|".join(k): {"val_score": v[0], "lr": v[1], "mu": v[2]} for k, v in best.items()}, fh, indent=1)
        jobs = []
        for preset in presets:
            for adapter, strats in STRATS.items():
                if adapter not in adapters:
                    continue
                for st in strats:
                    _, lr, mu = best[(preset, adapter, st)]
                    for seed in MAIN_SEEDS[:args.seeds]:
                        jobs.append(("main", preset, adapter, st, seed, lr, mu))

    have = done_keys(jl)
    todo = [j for j in jobs if key_of(*j) not in have]
    print(f"stage {args.stage}: {len(jobs)} runs, {len(jobs) - len(todo)} done, {len(todo)} to do", flush=True)
    batches = {}
    for j in todo:                                   # one batch per preset x adapter x strategy
        batches.setdefault(j[:4], []).append(j)
    t_start, done = time.time(), 0
    for (stage, preset, adapter, st), js in batches.items():
        if (time.time() - t_start) / 60 > args.budget_min:
            print(f"time budget reached; {len(todo) - done} runs left. Run the same command again.", flush=True)
            return
        t0 = time.time()
        meta, sp = data[preset]
        rounds, cps = (ROUNDS, CHECKPOINTS) if stage == "main" else (TUNE_ROUNDS, TUNE_CHECKPOINTS)
        outs, corrs = run_batch(meta, sp, adapter, st, [(j[4], j[5], j[6]) for j in js], rounds=rounds,
                                checkpoints=cps, keep=(stage == "main"))
        sec = (time.time() - t0) / len(js)
        with open(jl, "a") as fh:
            for j, res, corr in zip(js, outs, corrs):
                key = key_of(*j)
                rec = {"key": key, "stage": stage, "preset": preset, "adapter": adapter, "strategy": st,
                       "seed": j[4], "lr": j[5], "mu": j[6], "rounds": rounds, "local_epochs": LOCAL_EPOCHS,
                       "rank": RANK, "alpha": ALPHA, "tau": TAU, "lam": LAM, "rho": RHO, "sec": sec,
                       "checkpoints": {str(t): v for t, v in res["checkpoints"].items()},
                       "test_curve": res["test_curve"], "val_curve": res["val_curve"],
                       "energy_curve": res["energy_curve"] or None}
                if stage == "main":
                    fn = os.path.join(args.out, "runs", key.replace("|", "__") + ".npz")
                    payload = {f"correct_{t}_{c}": v for t, cl in corr.items() for c, v in enumerate(cl)}
                    payload["logs"] = np.array(json.dumps({str(t): v for t, v in res["logs"].items()}))
                    np.savez_compressed(fn, **payload)
                fh.write(json.dumps(rec) + "\n")
                done += 1
        accs_ = [o["checkpoints"][cps[-1]]["acc_eq"] for o in outs]
        print(f"[{done}/{len(todo)}] {stage} {preset} {adapter} {st}: {len(js)} runs, "
              f"test acc@{cps[-1]} {np.mean(accs_):.4f}, {sec:.0f} s/run", flush=True)
    print(f"stage {args.stage} complete", flush=True)


if __name__ == "__main__":
    main()
