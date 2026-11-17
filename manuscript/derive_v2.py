#!/usr/bin/env python3
"""Derived tables for the manuscript: validation-based early stopping and the validation-selected checkpoint.

Usage: python derive_v2.py <results_dir> <out_dir>
Writes es.csv, es_pairs.csv, cpsel.pkl, spec_gain_sel.csv and client_gain_sel.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

PLANNED = [("fedavg", "local"), ("fedprox", "local"), ("oracle", "local"), ("flexlora", "local"),
           ("ffa", "local"), ("fedsa", "local"), ("fedsaa", "local"), ("fedsaa_r0", "local"), ("fedsaa_u", "local"),
           ("centralized", "local"), ("fedsaa", "fedavg"), ("fedsaa_u", "fedavg"), ("fedsaa_u", "fedsaa"),
           ("fedsaa_r0", "fedsaa"), ("fedsaa_u", "fedsaa_r0"), ("fedsaa_u", "flexlora"), ("oracle", "fedavg"),
           ("fedsaa_u", "oracle"), ("fedprox", "fedavg")]
# the early-stopping family adds FlexLoRA - FedAvg to the 19 contrasts of fedbench_analyze.py
PLANNED_ES = PLANNED + [("flexlora", "fedavg")]
CPS = (40, 100, 200)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def paired(x, y):
    d = np.asarray(x) - np.asarray(y)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    p = stats.ttest_rel(x, y).pvalue
    pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return dict(diff=m, lo=m - h, hi=m + h, p=p, pw=pw, pos=int((d > 0).sum()))


def main():
    res, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    meta = {p: json.load(open(os.path.join(res, f"meta_{p}.json"))) for p in "AB"}
    runs = [json.loads(l) for l in open(os.path.join(res, "runs.jsonl"))]
    runs = [r for r in runs if r["stage"] == "main"]

    # ---- validation-based early stopping on the per-round curves
    rows = []
    for r in runs:
        v, t = np.asarray(r["val_curve"]), np.asarray(r["test_curve"])
        k = int(np.argmax(v))
        rows.append(dict(preset=r["preset"], adapter=r["adapter"], strategy=r["strategy"], seed=r["seed"],
                         es_round=k + 1, es_test=t[k] * 100, peak_test=t.max() * 100, t40=t[39] * 100,
                         t200=t[199] * 100))
    es = pd.DataFrame(rows).sort_values(["preset", "adapter", "strategy", "seed"])
    es.to_csv(os.path.join(out, "es.csv"), index=False)

    def vec(p, a, s):
        return es[(es.preset == p) & (es.adapter == a) & (es.strategy == s)].sort_values("seed").es_test.values

    # ---- planned contrasts at the early-stopped round
    P = []
    for p in "AB":
        for a in ("linear", "nonlinear"):
            fam = []
            for x, y in PLANNED_ES:
                if len(vec(p, a, x)) and len(vec(p, a, y)):
                    fam.append(dict(preset=p, adapter=a, a=x, b=y, **paired(vec(p, a, x), vec(p, a, y))))
            adj = holm([f["p"] for f in fam])
            for f, h in zip(fam, adj):
                f["holm"] = h
            P += fam
            P.append(dict(preset=p, adapter=a, a=f"{a} local", b="probe",
                          **paired(vec(p, a, "local"), vec(p, "none", "probe")), holm=np.nan))
        for s in ("local", "fedavg", "oracle", "centralized", "fedprox", "fedsa", "ffa"):
            P.append(dict(preset=p, adapter="nl-lin", a=f"nonlinear {s}", b=f"linear {s}",
                          **paired(vec(p, "nonlinear", s), vec(p, "linear", s)), holm=np.nan))
    pd.DataFrame(P).to_csv(os.path.join(out, "es_pairs.csv"), index=False)

    # ---- validation-selected checkpoint among the saved ones (40, 100, 200)
    C = []
    for r in runs:
        v = r["val_curve"]
        cp = max(CPS, key=lambda c: (v[c - 1], -c))
        m = r["checkpoints"][str(cp)]
        C.append(dict(preset=r["preset"], adapter=r["adapter"], strategy=r["strategy"], seed=r["seed"], cp=cp,
                      acc=m["acc_eq"] * 100, acc_w=m["acc_w"] * 100, f1=m["f1"], auc=m["auc"],
                      worst=m["worst"] * 100, per_client=m["per_client"]))
    cps = pd.DataFrame(C).sort_values(["preset", "adapter", "strategy", "seed"]).reset_index(drop=True)
    cps.to_pickle(os.path.join(out, "cpsel.pkl"))

    # ---- specialty and client gains over local-only at the selected checkpoint
    S, K = [], []
    for (p, a), g in cps[cps.adapter != "none"].groupby(["preset", "adapter"]):
        spec = np.asarray(meta[p]["client_specialty"])
        loc = np.stack(g[g.strategy == "local"].sort_values("seed").per_client.values) * 100
        for s in sorted(set(g.strategy) - {"local"}):
            x = np.stack(g[g.strategy == s].sort_values("seed").per_client.values) * 100
            d = x - loc                                   # seeds x clients
            for k, name in enumerate(meta[p]["specialties"]):
                ds = d[:, spec == k].mean(1)
                m, h = ds.mean(), stats.t.ppf(0.975, len(ds) - 1) * ds.std(ddof=1) / np.sqrt(len(ds))
                S.append(dict(preset=p, adapter=a, strategy=s, specialty=name, gain=m, lo=m - h, hi=m + h))
            dc = d.mean(0)
            K.append(dict(preset=p, adapter=a, strategy=s, above=int((dc > 0).sum()), below=int((dc < 0).sum()),
                          median=float(np.median(dc)), mn=dc.min(), mx=dc.max()))
    pd.DataFrame(S).to_csv(os.path.join(out, "spec_gain_sel.csv"), index=False)
    pd.DataFrame(K).to_csv(os.path.join(out, "client_gain_sel.csv"), index=False)
    print("wrote derived tables to", out)


if __name__ == "__main__":
    main()
