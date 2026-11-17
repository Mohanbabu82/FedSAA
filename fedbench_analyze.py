#!/usr/bin/env python3
"""Stage 3: every table, test and figure for the revised manuscript, from the raw run records.

Usage:
    python fedbench_analyze.py --feats feats --results results --out analysis

Writes CSV tables, PNG/PDF figures and summary.md to --out. Nothing is typed in by hand.
Seed-level tests: paired t-test, Wilcoxon signed-rank and exact sign-flip test over the 10 seeds, with
Holm correction within each family. Image-level tests: per-image correctness averaged over seeds,
client-equal-weight mean difference, stratified bootstrap CI and sign-flip randomization p value.
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

LABEL = {"local": "Local-only", "centralized": "Centralized", "fedavg": "FedAvg", "fedprox": "FedProx",
         "oracle": "Specialty oracle", "flexlora": "FlexLoRA", "ffa": "FFA-LoRA", "fedsa": "FedSA-LoRA",
         "fedsaa": "FedSAA", "fedsaa_r0": "FedSAA (rho = 0)", "fedsaa_u": "FedSAA-U", "probe": "Linear probe"}
PLANNED = [("fedavg", "local"), ("fedprox", "local"), ("oracle", "local"), ("flexlora", "local"),
           ("ffa", "local"), ("fedsa", "local"), ("fedsaa", "local"), ("fedsaa_r0", "local"), ("fedsaa_u", "local"),
           ("centralized", "local"), ("fedsaa", "fedavg"), ("fedsaa_u", "fedavg"), ("fedsaa_u", "fedsaa"),
           ("fedsaa_r0", "fedsaa"), ("fedsaa_u", "fedsaa_r0"), ("fedsaa_u", "flexlora"), ("oracle", "fedavg"),
           ("fedsaa_u", "oracle"), ("fedprox", "fedavg")]
METRICS = ["acc_eq", "acc_w", "f1", "auc", "worst"]


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    order = np.argsort(q)
    adj, run = np.empty(len(q)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(q) - rank) * q[i])
        adj[i] = min(1.0, run)
    out[ok] = adj
    return out


def signflip_p(d):
    d = np.asarray(d, float)
    n = len(d)
    if n <= 16:
        signs = np.array(list(itertools.product([1, -1], repeat=n)))
        stat = np.abs((signs * d).mean(1))
    else:
        rng = np.random.default_rng(0)
        stat = np.abs((rng.choice([1, -1], (20000, n)) * d).mean(1))
    return float(np.mean(stat >= abs(d.mean()) - 1e-12))


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    mu, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    if se > 0:
        tval = mu / se
        p_t = 2 * stats.t.sf(abs(tval), n - 1)
    else:
        tval, p_t = np.nan, (1.0 if mu == 0 else 0.0)
    crit = stats.t.ppf(0.975, n - 1)
    try:
        p_w = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p_w = np.nan
    return {"n": n, "diff": mu, "sd": sd, "ci_lo": mu - crit * se, "ci_hi": mu + crit * se, "t": tval,
            "p_t": p_t, "p_wilcoxon": p_w, "p_signflip": signflip_p(d), "same_sign": int(max((d > 0).sum(), (d < 0).sum()))}


def load(results):
    rows = [json.loads(l) for l in open(os.path.join(results, "runs.jsonl")) if l.strip()]
    main = [r for r in rows if r["stage"] == "main"]
    tune = [r for r in rows if r["stage"] == "tune"]
    return main, tune


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="feats")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    main_runs, tune_runs = load(args.results)
    meta = {}
    for f in sorted(glob.glob(os.path.join(args.results, "meta_*.json"))):
        meta[os.path.basename(f)[5:-5]] = json.load(open(f))
    if not meta:
        for f in sorted(glob.glob(os.path.join(args.feats, "preset_*.npz"))):
            meta[os.path.basename(f)[7:-4]] = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
    cps = sorted({int(t) for r in main_runs for t in r["checkpoints"]})
    R = lambda p, a, s: sorted([r for r in main_runs if (r["preset"], r["adapter"], r["strategy"]) == (p, a, s)],
                               key=lambda r: r["seed"])
    combos = sorted({(r["preset"], r["adapter"], r["strategy"]) for r in main_runs})
    W = lambda name: os.path.join(args.out, name)

    # ---------------- main table
    t_main = []
    for p, a, s in combos:
        rs = R(p, a, s)
        for t in cps:
            row = {"preset": p, "adapter": a, "strategy": s, "round": t, "n_seeds": len(rs),
                   "lr": rs[0]["lr"], "mu": rs[0]["mu"]}
            for m in METRICS:
                v = np.array([r["checkpoints"][str(t)][m] for r in rs]) * (100 if m in ("acc_eq", "acc_w", "worst") else 1)
                row[f"{m}_mean"], row[f"{m}_sd"] = v.mean(), v.std(ddof=1)
            t_main.append(row)
    t_main = pd.DataFrame(t_main)
    t_main.to_csv(W("T_main.csv"), index=False)

    # ---------------- seed-level contrasts
    t_pairs = []
    for p in sorted({c[0] for c in combos}):
        for a in sorted({c[1] for c in combos if c[0] == p}):
            strs = [c[2] for c in combos if c[:2] == (p, a)]
            if a == "none":
                continue
            for t in cps:
                for m in METRICS:
                    fam = []
                    for x, y in itertools.combinations(strs, 2):
                        if (y, x) in PLANNED:
                            x, y = y, x
                        sc = 100 if m in ("acc_eq", "acc_w", "worst") else 1
                        res = paired([r["checkpoints"][str(t)][m] * sc for r in R(p, a, x)],
                                     [r["checkpoints"][str(t)][m] * sc for r in R(p, a, y)])
                        res.update(preset=p, adapter=a, round=t, metric=m, a=x, b=y, planned=(x, y) in PLANNED)
                        fam.append(res)
                    ph = holm([f["p_t"] for f in fam])
                    pw = holm([f["p_wilcoxon"] for f in fam])
                    pl = [i for i, f in enumerate(fam) if f["planned"]]
                    php = holm([fam[i]["p_t"] for i in pl])
                    for i, f in enumerate(fam):
                        f["p_t_holm_all"], f["p_w_holm_all"] = ph[i], pw[i]
                        f["p_t_holm_planned"] = php[pl.index(i)] if i in pl else np.nan
                    t_pairs.extend(fam)
        # adapter-level contrasts (Proposition 1): linear and nonlinear adapters against the linear probe
        for t in cps:
            for x in ("linear", "nonlinear"):
                if R(p, x, "local") and R(p, "none", "probe"):
                    res = paired([r["checkpoints"][str(t)]["acc_eq"] * 100 for r in R(p, x, "local")],
                                 [r["checkpoints"][str(t)]["acc_eq"] * 100 for r in R(p, "none", "probe")])
                    res.update(preset=p, adapter=f"{x} vs none", round=t, metric="acc_eq", a=f"{x}:local",
                               b="none:probe", planned=True)
                    t_pairs.append(res)
            for s in ("local", "fedavg", "oracle", "centralized"):
                if R(p, "linear", s) and R(p, "nonlinear", s):
                    res = paired([r["checkpoints"][str(t)]["acc_eq"] * 100 for r in R(p, "nonlinear", s)],
                                 [r["checkpoints"][str(t)]["acc_eq"] * 100 for r in R(p, "linear", s)])
                    res.update(preset=p, adapter="nonlinear vs linear", round=t, metric="acc_eq",
                               a=f"nonlinear:{s}", b=f"linear:{s}", planned=True)
                    t_pairs.append(res)
    t_pairs = pd.DataFrame(t_pairs)
    t_pairs.to_csv(W("T_pairs_seed.csv"), index=False)

    # ---------------- image-level contrasts (planned only)
    def correct(run, t, C):
        fn = os.path.join(args.results, "runs", run["key"].replace("|", "__") + ".npz")
        z = np.load(fn, allow_pickle=False)
        return [z[f"correct_{t}_{c}"].astype(float) for c in range(C)]

    t_img = []
    rng = np.random.default_rng(0)
    for p in sorted({c[0] for c in combos}):
        C = meta[p]["n_clients"]
        for a in ("linear", "nonlinear"):
            for x, y in PLANNED:
                rx, ry = R(p, a, x), R(p, a, y)
                if not rx or not ry:
                    continue
                for t in cps:
                    cx = [correct(r, t, C) for r in rx]
                    cy = [correct(r, t, C) for r in ry]
                    dcl = [np.mean([s[c] for s in cx], 0) - np.mean([s[c] for s in cy], 0) for c in range(C)]
                    stat = np.mean([d.mean() for d in dcl]) * 100
                    boot = np.array([np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in dcl])
                                     for _ in range(1000)]) * 100
                    flips = np.array([np.mean([(d * rng.choice([-1, 1], len(d))).mean() for d in dcl])
                                      for _ in range(1000)]) * 100
                    wdiff = np.concatenate(dcl).mean() * 100
                    t_img.append({"preset": p, "adapter": a, "round": t, "a": x, "b": y, "diff_eq": stat,
                                  "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
                                  "p_signflip_images": float((np.abs(flips) >= abs(stat) - 1e-12).mean()),
                                  "diff_size_weighted": wdiff})
    t_img = pd.DataFrame(t_img)
    if len(t_img):
        for (p, a, t), g in t_img.groupby(["preset", "adapter", "round"]):
            t_img.loc[g.index, "p_holm"] = holm(g.p_signflip_images.values)
    t_img.to_csv(W("T_pairs_image.csv"), index=False)

    # ---------------- per-specialty and per-client gains over local
    t_spec, t_cli = [], []
    for p, a, s in combos:
        if s in ("local", "probe"):
            continue
        rl, rs = R(p, a, "local"), R(p, a, s)
        sp = np.array(meta[p]["client_specialty"])
        for t in cps:
            loc = np.array([r["checkpoints"][str(t)]["per_client"] for r in rl]) * 100
            cur = np.array([r["checkpoints"][str(t)]["per_client"] for r in rs]) * 100
            for g, name in enumerate(meta[p]["specialties"]):
                d = cur[:, sp == g].mean(1) - loc[:, sp == g].mean(1)
                res = paired(d, np.zeros_like(d))
                t_spec.append({"preset": p, "adapter": a, "strategy": s, "round": t, "specialty": name,
                               "gain": res["diff"], "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"], "p_t": res["p_t"]})
            dc = (cur - loc).mean(0)
            t_cli.append({"preset": p, "adapter": a, "strategy": s, "round": t, "clients_above": int((dc > 0).sum()),
                          "clients_below": int((dc < 0).sum()), "median_gain": float(np.median(dc)),
                          "min_gain": float(dc.min()), "max_gain": float(dc.max())})
    pd.DataFrame(t_spec).to_csv(W("T_specialty_gain.csv"), index=False)
    pd.DataFrame(t_cli).to_csv(W("T_client_gain.csv"), index=False)

    # ---------------- convergence
    t_conv = []
    for p, a, s in combos:
        rs = R(p, a, s)
        cur = np.array([r["test_curve"] for r in rs]) * 100
        T = cur.shape[1]
        mean = cur.mean(0)
        last = min(20, T)
        slope = np.mean([np.polyfit(np.arange(T - last, T) + 1, c[-last:], 1)[0] for c in cur])
        within = int(np.argmax(np.abs(mean - mean[-1]) <= 0.5)) + 1
        t_conv.append({"preset": p, "adapter": a, "strategy": s, "final": mean[-1], "peak": mean.max(),
                       "peak_round": int(mean.argmax()) + 1, "slope_last20_per_round": slope,
                       "first_round_within_0.5_of_final": within,
                       **{f"r{k}": mean[k - 1] for k in (10, 20, 40, 100, 150, 200) if k <= T}})
    pd.DataFrame(t_conv).to_csv(W("T_convergence.csv"), index=False)

    # ---------------- measured similarities, attention shares and SVD energy
    t_sim = []
    for p, a, s in combos:
        if not s.startswith("fedsaa") and s != "flexlora":
            continue
        sp = np.array(meta[p]["client_specialty"])
        same = (sp[:, None] == sp[None, :]) & ~np.eye(len(sp), dtype=bool)
        other = sp[:, None] != sp[None, :]
        acc = {}
        for r in R(p, a, s):
            fn = os.path.join(args.results, "runs", r["key"].replace("|", "__") + ".npz")
            logs = json.loads(str(np.load(fn, allow_pickle=False)["logs"]))
            for t, lg in logs.items():
                e = acc.setdefault(int(t), {k: [] for k in ("S_same", "S_other", "w_self", "w_same", "w_other", "energy")})
                if "S" in lg:
                    S, Wt = np.array(lg["S"]), np.array(lg["Wt"])
                    e["S_same"].append(S[same].mean()); e["S_other"].append(S[other].mean())
                    e["w_self"].append(np.diag(Wt).mean())
                    e["w_same"].append((Wt * same).sum(1).mean()); e["w_other"].append((Wt * other).sum(1).mean())
                if "energy" in lg:
                    e["energy"].append(float(np.mean(lg["energy"])))
        for t, e in sorted(acc.items()):
            t_sim.append({"preset": p, "adapter": a, "strategy": s, "round": t,
                          **{k: (float(np.mean(v)) if v else np.nan) for k, v in e.items()}})
    pd.DataFrame(t_sim).to_csv(W("T_similarity_energy.csv"), index=False)

    # ---------------- tuning record
    t_tune = pd.DataFrame([{"preset": r["preset"], "adapter": r["adapter"], "strategy": r["strategy"], "lr": r["lr"],
                            "mu": r["mu"], **{f"val_r{t}": v["val_acc_eq"] * 100 for t, v in r["checkpoints"].items()}}
                           for r in tune_runs])
    t_tune.to_csv(W("T_tuning.csv"), index=False)
    sel = os.path.join(args.results, "selected_configs.json")
    if os.path.exists(sel):
        with open(sel) as fh, open(W("selected_configs.json"), "w") as gh:
            gh.write(fh.read())

    figures(main_runs, combos, meta, cps, args.out, pd.DataFrame(t_spec))
    summary(t_main, t_pairs, t_img, pd.DataFrame(t_conv), pd.DataFrame(t_sim), cps, W("summary.md"))
    print("analysis written to", args.out)


def figures(main_runs, combos, meta, cps, out, t_spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    col = {"local": "#0072B2", "centralized": "#000000", "fedavg": "#E69F00", "fedprox": "#CC79A7",
           "oracle": "#009E73", "flexlora": "#56B4E9", "ffa": "#999999", "fedsa": "#F0E442",
           "fedsaa": "#D55E00", "fedsaa_r0": "#8C2D04", "fedsaa_u": "#7B3294", "probe": "#666666"}
    presets = sorted({c[0] for c in combos})
    for a in ("linear", "nonlinear"):
        fig, axes = plt.subplots(len(presets), 1, figsize=(3.4, 2.2 * len(presets)), sharex=True, squeeze=False)
        for ax, p in zip(axes[:, 0], presets):
            for (pp, aa, s) in combos:
                if (pp, aa) != (p, a):
                    continue
                cur = np.array([r["test_curve"] for r in main_runs if (r["preset"], r["adapter"], r["strategy"], r["stage"]) == (p, a, s, "main")]) * 100
                ax.plot(np.arange(1, cur.shape[1] + 1), cur.mean(0), color=col[s], lw=1.0, label=LABEL[s])
            ax.set_title(f"Preset {p}, {a} adapter", loc="left", fontsize=7.5, fontweight="bold")
            ax.set_ylabel("Test accuracy (%)")
            ax.grid(axis="y", lw=0.3, alpha=0.5)
        axes[-1, 0].set_xlabel("Communication round")
        axes[0, 0].legend(frameon=False, fontsize=5.5, ncol=2)
        fig.savefig(os.path.join(out, f"fig_curves_{a}.png"), dpi=600, bbox_inches="tight")
        fig.savefig(os.path.join(out, f"fig_curves_{a}.pdf"), bbox_inches="tight")
        plt.close(fig)
    if len(t_spec):
        t = max(cps)
        sub = t_spec[(t_spec["round"] == t) & (t_spec.adapter == "linear") &
                     t_spec.strategy.isin(["fedavg", "oracle", "fedsaa", "fedsaa_u"])]
        names = list(dict.fromkeys(sub.preset + ":" + sub.specialty))
        fig, ax = plt.subplots(figsize=(3.4, 3.6))
        off = {"fedavg": 0.24, "oracle": 0.08, "fedsaa": -0.08, "fedsaa_u": -0.24}
        for s, g in sub.groupby("strategy"):
            y = [-names.index(n) + off[s] for n in g.preset + ":" + g.specialty]
            ax.errorbar(g.gain, y, xerr=[g.gain - g.ci_lo, g.ci_hi - g.gain], fmt="D", ms=3, lw=0.7,
                        color=col[s], label=LABEL[s])
        ax.axvline(0, color="k", lw=0.7)
        ax.set_yticks([-i for i in range(len(names))])
        ax.set_yticklabels([n.split(":")[1] + f" ({n.split(':')[0]})" for n in names])
        ax.set_xlabel(f"Gain over local-only at round {t} (points), mean and 95% CI")
        ax.legend(frameon=False, fontsize=5.5)
        fig.savefig(os.path.join(out, "fig_specialty_gain.png"), dpi=600, bbox_inches="tight")
        fig.savefig(os.path.join(out, "fig_specialty_gain.pdf"), bbox_inches="tight")
        plt.close(fig)


def summary(t_main, t_pairs, t_img, t_conv, t_sim, cps, path):
    L = ["# Revised experiments: summary", ""]
    for t in cps:
        L.append(f"## Accuracy at round {t} (mean ± SD over seeds, equal-weight / size-weighted)")
        sub = t_main[t_main["round"] == t]
        for _, r in sub.iterrows():
            L.append(f"- {r.preset} {r.adapter:9} {LABEL[r.strategy]:18} {r.acc_eq_mean:6.2f} ± {r.acc_eq_sd:4.2f}"
                     f" / {r.acc_w_mean:6.2f}  F1 {r.f1_mean:.3f}  AUC {r.auc_mean:.4f}  worst {r.worst_mean:5.2f}"
                     f"  (lr {r.lr:g}, mu {r.mu:g})")
        L.append("")
    if len(t_pairs):
        L.append("## Planned seed-level contrasts, accuracy (Holm within the planned family)")
        sub = t_pairs[(t_pairs.metric == "acc_eq") & t_pairs.planned]
        for _, r in sub.iterrows():
            L.append(f"- r{r['round']} {r.preset} {r.adapter}: {r.a} − {r.b} = {r['diff']:+.2f} "
                     f"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}], p_t {r.p_t:.4f}, Holm {r.p_t_holm_planned:.4f}, "
                     f"Wilcoxon {r.p_wilcoxon:.4f}, same sign {r.same_sign}/{r.n}")
        L.append("")
    if len(t_img):
        L.append("## Image-level contrasts (per-image correctness averaged over seeds)")
        for _, r in t_img.iterrows():
            L.append(f"- r{r['round']} {r.preset} {r.adapter}: {r.a} − {r.b} = {r.diff_eq:+.2f} "
                     f"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}], p {r.p_signflip_images:.3f}, Holm {r.p_holm:.3f}")
        L.append("")
    L.append("## Convergence")
    for _, r in t_conv.iterrows():
        L.append(f"- {r.preset} {r.adapter} {LABEL[r.strategy]}: final {r.final:.2f}, peak {r.peak:.2f} at round "
                 f"{r.peak_round}, slope over last 20 rounds {r.slope_last20_per_round:+.4f}/round")
    if len(t_sim):
        L.append("")
        L.append("## Measured similarities, attention shares and retained SVD energy")
        for _, r in t_sim.iterrows():
            L.append(f"- {r.preset} {r.adapter} {r.strategy} r{r['round']}: S same {r.S_same:.3f}, S other "
                     f"{r.S_other:.3f}, w self {r.w_self:.3f}, same {r.w_same:.3f}, other {r.w_other:.3f}, "
                     f"energy {r.energy:.3f}")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
