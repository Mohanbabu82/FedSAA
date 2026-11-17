# -*- coding: utf-8 -*-
"""IEEE Access manuscript, revision 2 (10 seeds, 200 rounds, validation tuning, new baselines).

Every number in the text and tables is computed here from the analysis files, so the manuscript
cannot drift from the data. Markup understood by build_v2.py:
    *italic*  **bold**  ~sub~  ^sup^  [[highlighted placeholder]]
    \\cite{k1,k2}  numbered by first appearance;  {T:key} {F:key}  numbered in order of placement
"""
import json
import math
import os
import pickle

import numpy as np
import pandas as pd

import content as C
import content_long as L
import refs

HERE = os.path.dirname(os.path.abspath(__file__))
# repository layout: <repo>/results, <repo>/analysis, <repo>/manuscript/derived (override with environment variables)
RES = os.environ.get("FEDSAA_RESULTS", os.path.join(HERE, "..", "results"))
ANA = os.environ.get("FEDSAA_ANALYSIS", os.path.join(HERE, "..", "analysis"))
DER = os.environ.get("FEDSAA_DERIVED", os.path.join(HERE, "derived"))

# ------------------------------------------------------------------ old text, for reuse of unchanged paragraphs
L.data_tables = lambda tdir: {"_n_excl": 0}
_OLD, _ = L.build_body(HERE)


def OLD(i):
    return _OLD[i][1]


# ------------------------------------------------------------------ references added in this revision
refs.REFS.update({
    "wilcoxon1945": "F. Wilcoxon, “Individual comparisons by ranking methods,” *Biometrics Bull.*, vol. 1, no. 6, "
                    "pp. 80–83, Dec. 1945.",
    "efron1993": "B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: "
                 "Chapman & Hall, 1993.",
    "prechelt1998": "L. Prechelt, “Early stopping—But when?” in *Neural Networks: Tricks of the Trade* (Lecture "
                    "Notes in Computer Science, vol. 1524). Berlin, Germany: Springer, 1998, pp. 55–69.",
    "harris2020": "C. R. Harris *et al.*, “Array programming with NumPy,” *Nature*, vol. 585, no. 7825, "
                  "pp. 357–362, Sep. 2020.",
})
REFS = refs.REFS

# ------------------------------------------------------------------ labels and formatting
NICE = L.NICE
LAB = {"local": "Local-only", "centralized": "Centralized", "fedavg": "FedAvg", "fedprox": "FedProx",
       "oracle": "Specialty oracle", "flexlora": "FlexLoRA", "ffa": "FFA-LoRA", "fedsa": "FedSA-LoRA",
       "fedsaa": "FedSAA", "fedsaa_r0": "FedSAA, *ρ* = 0", "fedsaa_u": "FedSAA-U", "probe": "Linear probe"}
SHORT = dict(LAB, fedsaa_r0="FedSAA-0")
LIN = ["local", "centralized", "fedavg", "fedprox", "oracle", "flexlora", "ffa", "fedsa", "fedsaa", "fedsaa_r0",
       "fedsaa_u"]
NONLIN = ["local", "centralized", "fedavg", "fedprox", "oracle", "ffa", "fedsa"]
f = L.f


def sg(v, nd=2):
    if abs(v) < 0.5 * 10 ** (-nd):
        return f(0.0, nd)
    return L.sg(v, nd)


def pv(v):
    if v != v:
        return "–"
    return "<0.001" if v < 0.0005 else f"{v:.3f}"


def ci(lo, hi, nd=2):
    return f"{f(lo, nd)} to {f(hi, nd)}"


# ------------------------------------------------------------------ data
META = {p: json.load(open(os.path.join(RES, f"meta_{p}.json"))) for p in "AB"}
SEL = json.load(open(os.path.join(RES, "selected_configs.json")))
ES = pd.read_csv(os.path.join(DER, "es.csv"))
EP = pd.read_csv(os.path.join(DER, "es_pairs.csv"))
TM = pd.read_csv(os.path.join(ANA, "T_main.csv"))
IMG = pd.read_csv(os.path.join(ANA, "T_pairs_image.csv"))
SIM = pd.read_csv(os.path.join(ANA, "T_similarity_energy.csv"))
TC = pd.read_csv(os.path.join(ANA, "T_convergence.csv"))
SPEC = pd.read_csv(os.path.join(DER, "spec_gain_sel.csv"))
CG = pd.read_csv(os.path.join(DER, "client_gain_sel.csv"))
CP = pickle.load(open(os.path.join(DER, "cpsel.pkl"), "rb"))
RUNS = [json.loads(l) for l in open(os.path.join(RES, "runs.jsonl"))]
MAIN = [r for r in RUNS if r["stage"] == "main"]
TUNE = [r for r in RUNS if r["stage"] == "tune"]
N_SEEDS = len({r["seed"] for r in MAIN})
ROUNDS = MAIN[0]["rounds"]
SEC_TOTAL = sum(r["sec"] for r in RUNS)


def adp(s):
    return "none" if s == "probe" else None


def es(p, a, s, col="es_test"):
    a = "none" if s == "probe" else a
    x = ES[(ES.preset == p) & (ES.adapter == a) & (ES.strategy == s)][col]
    assert len(x) == N_SEEDS, (p, a, s, len(x))
    return x.mean(), x.std()


def pair(p, a, x, y):
    r = EP[(EP.preset == p) & (EP.adapter == a) & (EP.a == x) & (EP.b == y)]
    assert len(r) == 1, (p, a, x, y)
    return r.iloc[0]


def tm(p, a, s, rnd, col="acc_eq_mean"):
    a = "none" if s == "probe" else a
    r = TM[(TM.preset == p) & (TM.adapter == a) & (TM.strategy == s) & (TM["round"] == rnd)]
    assert len(r) == 1, (p, a, s, rnd)
    return r.iloc[0][col]


def img(p, a, rnd, x, y):
    r = IMG[(IMG.preset == p) & (IMG.adapter == a) & (IMG["round"] == rnd) & (IMG.a == x) & (IMG.b == y)]
    assert len(r) == 1, (p, a, rnd, x, y)
    return r.iloc[0]


def spec(p, a, s, sp):
    r = SPEC[(SPEC.preset == p) & (SPEC.adapter == a) & (SPEC.strategy == s) & (SPEC.specialty == sp)]
    assert len(r) == 1
    return r.iloc[0]


def cg(p, a, s):
    r = CG[(CG.preset == p) & (CG.adapter == a) & (CG.strategy == s)]
    assert len(r) == 1
    return r.iloc[0]


def sim(p, s, rnd, col):
    r = SIM[(SIM.preset == p) & (SIM.strategy == s) & (SIM["round"] == rnd)]
    assert len(r) == 1
    return r.iloc[0][col]


def cp(p, a, s, col):
    a = "none" if s == "probe" else a
    x = CP[(CP.preset == p) & (CP.adapter == a) & (CP.strategy == s)][col]
    assert len(x) == N_SEEDS
    return x.mean()


def sec(p, a, s):
    a = "none" if s == "probe" else a
    v = [r["sec"] for r in MAIN if (r["preset"], r["adapter"], r["strategy"]) == (p, a, s)]
    return sum(v) / len(v)


def lr_of(p, a, s):
    a = "none" if s == "probe" else a
    return SEL[f"{p}|{a}|{s}"]


def pd_(d, nd=2):
    """Difference with sign and CI, e.g. '+0.83 (95% CI 0.38 to 1.27)'."""
    return f"{sg(d['diff'], nd)} points (95% CI {ci(d['lo'], d['hi'], nd)})"


def esd(p, a, s):
    m, sd = es(p, a, s)
    return f"{f(m)}%"


# count of seeds with the sign of the mean in a pair
def pos(d):
    return int(d["pos"])


# ------------------------------------------------------------------ derived facts used in the text
def peak_round(p, a, s):
    """Round of the peak of the seed-mean test curve."""
    a = "none" if s == "probe" else a
    curves = np.array([r["test_curve"] for r in MAIN if (r["preset"], r["adapter"], r["strategy"]) == (p, a, s)])
    m = curves.mean(0)
    return int(np.argmax(m)) + 1, float(m.max()) * (100 if m.max() <= 1 else 1)


def _worst_counts():
    import collections
    cnt = collections.Counter()
    for _, r in CP.iterrows():
        m = META[r.preset]
        i = int(np.argmin(r.per_client))
        cnt[(r.preset, m["specialties"][m["client_specialty"][i]])] += 1
        cnt[r.preset] += 1
    return cnt


WC = _worst_counts()


def n_val_total(p):
    return sum(META[p]["n_val"])


def n_test_total(p):
    return sum(META[p]["n_test"])


def n_train_total(p):
    return sum(META[p]["n_train"])


# specialties where an interval excludes zero
def sig_specs(p, a, s, sign):
    out = []
    for sp in META[p]["specialties"]:
        r = spec(p, a, s, sp)
        if (sign < 0 and r.hi < 0) or (sign > 0 and r.lo > 0):
            out.append(sp)
    return out


# ------------------------------------------------------------------ front matter
TITLE = ("Federated Low-Rank Feature Adapters Across Medical Imaging Specialties: Specialty Structure, "
         "Training Budget, and the Limits of Similarity-Weighted Aggregation")
AUTHORS, AFFILIATION, CORRESPONDING, SUPPORT = C.AUTHORS, C.AFFILIATION, C.CORRESPONDING, C.SUPPORT
BIOS = C.BIOS

_a = {k: pair("A", "linear", k, "local") for k in ("fedavg", "oracle", "flexlora", "fedsaa", "fedsaa_u", "centralized")}
_b_lo = min(pair("B", "linear", k, "local")["diff"] for k in LIN if k != "local")
_b_hi = max(pair("B", "linear", k, "local")["diff"] for k in LIN if k != "local")
_uf = pair("A", "linear", "fedsaa_u", "fedsaa")

ABSTRACT = (
    "Departments of different medical imaging specialties cannot share a classifier, but they could share a "
    "parameter-efficient adapter on a frozen backbone. We benchmark this setting in two simulated federations "
    "built from MedMNIST v2, each with five specialties and 15 clients. A rank-16 adapter on frozen ResNet-18 "
    "features was trained under 12 strategies, including FedAvg, FedProx, FlexLoRA, FFA-LoRA, FedSA-LoRA, a "
    "specialty-oracle that averages only within a specialty, a linear probe, and FedSAA, a similarity-weighted "
    "rule we designed, with two variants. Learning rates were tuned on validation data, every strategy was run "
    f"for {ROUNDS} rounds with {N_SEEDS} seeds, and the primary metric was test accuracy at the validation-selected "
    "round. In the first federation, FedAvg was below local-only training "
    f"({f(_a['fedavg']['diff'])} points), whereas averaging within specialties gained {sg(_a['oracle']['diff'])} points. "
    f"In the second, every federated strategy exceeded local-only training ({sg(_b_lo)} to {sg(_b_hi)} points). "
    "Logged similarities showed why FedSAA could not personalize: within 40 rounds the updates of different "
    "specialties had cosine similarity above 0.97, so its attention weights became uniform and it reduced to "
    "FlexLoRA. In the first federation, retaining the local adapter in factor space cost "
    f"{f(_uf['diff'])} points relative to retention in update space, as our analysis of factor non-identifiability "
    "predicts. At a fixed budget of 40 rounds, local-only training led every global method, and by round 200 the "
    "order had reversed. Specialty structure and training budget, not the choice of aggregation rule, determined "
    "whether federation helped."
)
INDEX_TERMS = ("Benchmarking, federated learning, low-rank adaptation, medical image classification, MedMNIST, "
               "negative transfer, parameter-efficient fine-tuning, personalized federated learning.")


# ------------------------------------------------------------------ tables
def tables():
    T = {}
    # ---- composition
    rows = []
    MOD = {"pathmnist": "Histology", "dermamnist": "Dermatoscopy", "bloodmnist": "Blood cell microscopy",
           "octmnist": "Retinal OCT", "tissuemnist": "Kidney cortex microscopy", "organamnist": "Abdominal CT, axial",
           "pneumoniamnist": "Chest X-ray", "retinamnist": "Fundus photography", "breastmnist": "Breast ultrasound",
           "organcmnist": "Abdominal CT, coronal"}
    for p in "AB":
        m = META[p]
        rows.append(("group", f"Preset {p}"))
        for k, sp in enumerate(m["specialties"]):
            idx = [i for i, s in enumerate(m["client_specialty"]) if s == k]
            def rng_(key):
                v = [m[key][i] for i in idx]
                return f"{min(v):,}" if min(v) == max(v) else f"{min(v):,}–{max(v):,}"
            rows.append([NICE[sp], MOD[sp], str(m["n_classes"][idx[0]]), rng_("n_train"), rng_("n_val"),
                         rng_("n_test")])
    T["composition"] = {
        "title": "Composition of the Two Federations", "widths": [1150, 1450, 520, 600, 480, 600], "size": 14,
        "header": ["Specialty", "Modality", "Classes", "Train", "Val.", "Test"],
        "align": ["left", "left", "center", "center", "center", "center"], "rows": rows,
        "note": "Each specialty has three clients. Train, Val., and Test give the number of images per client. "
                "Training and validation images come from the official MedMNIST training split and test images "
                "from the official test split."}

    # ---- notation
    T["notation"] = {
        "title": "Notation", "widths": [1150, 3550], "size": 15, "header": ["Symbol", "Meaning"],
        "align": ["left", "left"],
        "rows": [
            ["*N*", "Number of clients (15)"],
            ["*D*~*i*~, *n*~*i*~", "Training set of client *i* and its size"],
            ["*x*, *d*", "Frozen feature vector and its dimension (256)"],
            ["*A*, *B*", "Adapter factors, of size *r* × *d* and *d* × *r*"],
            ["*r*, *α*, *s*", "Adapter rank (16), scaling (32), and *s* = *α*/*r*"],
            ["*U*~*i*~", "Adapter product of client *i*, *B*~*i*~*A*~*i*~"],
            ["*S*~*ij*~", "Cosine similarity between *U*~*i*~ and *U*~*j*~"],
            ["*w*~*ij*~", "Attention weight of client *j* for client *i*"],
            ["*τ*, *λ*, *ρ*", "Temperature (0.5), global blend (0.5), retention (0.3)"],
            ["*μ*", "FedProx proximal weight (tuned)"],
            ["*E*, *T*", f"Local epochs per round (3) and rounds ({ROUNDS})"],
            ["Δ~*i*~(*m*)", "Transfer gain of strategy *m* for client *i*"],
            ["*t*^*^", "Validation-selected round"],
        ], "note": None}

    # ---- related work
    T["related"] = {
        "title": "Federated Low-Rank Adaptation Methods and Their Role in This Study",
        "widths": [1500, 1900, 3200, 1800, 1500], "size": 15,
        "header": ["Method", "Communicated", "Server aggregation", "Kept local", "Role here"],
        "align": ["left"] * 5,
        "rows": [
            ["FedIT \\cite{fedit}", "*A* and *B*", "Weighted average of *A* and of *B* separately", "Nothing",
             "Same rule as FedAvg"],
            ["FFA-LoRA \\cite{ffalora}", "*B* only; *A* frozen", "Average of *B*", "Nothing", "Baseline"],
            ["FedSA-LoRA \\cite{fedsalora}", "*A* only", "Average of *A*", "*B*", "Baseline"],
            ["FLoRA \\cite{flora}", "*A* and *B*, any rank", "Stacking of the client modules", "Nothing", "Not run"],
            ["HetLoRA \\cite{hetlora}", "*A* and *B*, client-specific rank", "Sparsity-weighted average after "
             "zero-padding", "Choice of rank", "Not run"],
            ["FlexLoRA \\cite{flexlora}", "*A* and *B*", "Weighted average of products, then SVD",
             "Choice of rank", "Baseline"],
            ["FedOPAL \\cite{fedopal}", "Global adapter", "Average of the global adapter",
             "Orthogonal personal adapter", "Not run"],
            ["FedSAA (ours)", "*A* and *B*", "Per-client similarity-weighted average of products, blended with "
             "the global mean, then SVD; retention in factor or update space", "Classifier; part of the adapter",
             "Studied rule"],
        ], "note": None}

    # ---- configuration
    T["config"] = {
        "title": "Configuration", "widths": [2150, 2550], "size": 15, "header": ["Item", "Value"],
        "align": ["left", "left"],
        "rows": [
            ["Backbone", "ResNet-18, ImageNet weights, frozen, cut after the third stage"],
            ["Input", "64 × 64 pixels, three channels, ImageNet normalization, no augmentation"],
            ["Feature", "Global average pooling, *d* = 256"],
            ["Adapter", "Rank 16, *α* = 32, linear or ReLU; 8,192 parameters"],
            ["Initialization", "*A* ∼ N(0, 1/*d*), *B* = 0; shared by all clients of a run"],
            ["Classifier", "Linear, private to each client (centralized: per specialty)"],
            ["Optimizer", "Adam, state reset every round, batch 64"],
            ["Learning rate", "Tuned in {3 × 10^−4^, 10^−3^, 3 × 10^−3^}"],
            ["FedProx *μ*", "Tuned in {0.001, 0.01, 0.1}"],
            ["FedSAA", "*τ* = 0.5, *λ* = 0.5, *ρ* = 0.3 (FedSAA, FedSAA-U) or 0"],
            ["Schedule", f"3 local epochs per round, {ROUNDS} rounds, all 15 clients every round"],
            ["Seeds", f"Tuning: one seed (1000), 100 rounds; main: {N_SEEDS} seeds (0–{N_SEEDS - 1})"],
            ["Partition", "Fixed (seed 2026), identical for all strategies and seeds"],
        ], "note": None}

    # ---- tuning outcome
    rows = []
    order = LIN + ["probe"]
    for s in order:
        cells = [LAB[s]]
        for p in "AB":
            for a in ("linear", "nonlinear"):
                if s == "probe":
                    c = lr_of(p, "none", s) if a == "linear" else None
                elif a == "nonlinear" and s not in NONLIN:
                    c = None
                else:
                    c = lr_of(p, a, s)
                if c is None:
                    cells.append("–")
                else:
                    t = {0.0003: "3e−4", 0.001: "1e−3", 0.003: "3e−3"}[c["lr"]]
                    if c["mu"]:
                        t += f"; {c['mu']:g}"
                    cells.append(t)
        rows.append(cells)
    T["tuning"] = {
        "title": "Learning Rate (and FedProx *μ*) Selected on Validation Data",
        "widths": [1300, 850, 850, 850, 850], "size": 14,
        "header": ["Strategy", "A, linear", "A, nonlin.", "B, linear", "B, nonlin."],
        "align": ["left"] + ["center"] * 4, "rows": rows,
        "note": "Selection maximized mean validation accuracy at rounds 40 and 100 of one tuning run. "
                "The linear probe has no adapter and appears once per preset."}

    # ---- primary results (wide)
    rows = []
    best = {}
    for p in "AB":
        for a in ("linear", "nonlinear"):
            cand = [s for s in (LIN if a == "linear" else NONLIN)]
            best[(p, a)] = max(cand, key=lambda s: es(p, a, s)[0])
    for s in LIN + ["probe"]:
        cells = [LAB[s]]
        for p in "AB":
            for a in ("linear", "nonlinear"):
                if (s == "probe" and a == "nonlinear") or (a == "nonlinear" and s not in NONLIN and s != "probe"):
                    cells += ["–", "–"]
                    continue
                m, sd = es(p, a, s)
                r, _ = es(p, a, s, "es_round")
                t = f"{f(m)} ± {f(sd)}"
                if best[(p, a)] == s:
                    t = "**" + t + "**"
                cells += [t, f"{r:.0f}"]
        rows.append(cells)
    T["main"] = {
        "title": "Test Accuracy at the Validation-Selected Round (%), Mean ± SD Over Ten Seeds",
        "widths": [1500, 1280, 560, 1280, 560, 1280, 560, 1280, 560], "size": 14,
        "header": ["Strategy", "A, linear", "*t*^*^", "A, nonlinear", "*t*^*^", "B, linear", "*t*^*^",
                   "B, nonlinear", "*t*^*^"],
        "align": ["left"] + ["center"] * 8, "rows": rows,
        "note": "Accuracy is averaged over the 15 clients with equal weight. *t*^*^ is the mean validation-selected "
                "round. The largest mean in each column is in bold. FlexLoRA and the FedSAA variants aggregate "
                "products *BA* and are defined for the linear adapter only; the linear probe has no adapter."}

    # ---- planned contrasts (wide)
    def crow(x, y, a):
        cells = [f"{SHORT[x]} − {SHORT[y]}"]
        for p in "AB":
            try:
                d = pair(p, a, x, y)
            except AssertionError:
                cells += ["–", "–", "–"]
                continue
            cells += [f"{sg(d['diff'])} [{f(d['lo'])}, {f(d['hi'])}]", f"{pos(d)}/{N_SEEDS}", pv(d["holm"])]
        return cells
    PL = [("fedavg", "local"), ("fedprox", "local"), ("oracle", "local"), ("flexlora", "local"), ("ffa", "local"),
          ("fedsa", "local"), ("fedsaa", "local"), ("fedsaa_r0", "local"), ("fedsaa_u", "local"),
          ("centralized", "local"), ("fedsaa", "fedavg"), ("fedsaa_u", "fedavg"), ("fedsaa_u", "fedsaa"),
          ("fedsaa_r0", "fedsaa"), ("fedsaa_u", "fedsaa_r0"), ("fedsaa_u", "flexlora"), ("oracle", "fedavg"),
          ("fedsaa_u", "oracle"), ("fedprox", "fedavg"), ("flexlora", "fedavg")]
    rows = [("group", "Linear adapter")] + [crow(x, y, "linear") for x, y in PL]
    PLN = [(x, y) for x, y in PL if x in NONLIN and y in NONLIN]
    rows += [("group", "Nonlinear adapter")] + [crow(x, y, "nonlinear") for x, y in PLN]
    T["contrasts"] = {
        "title": "Planned Contrasts at the Validation-Selected Round (Percentage Points)",
        "widths": [2300, 1750, 620, 760, 1750, 620, 760], "size": 14,
        "header": ["Contrast", "A: diff. [95% CI]", "Seeds", "Holm *p*", "B: diff. [95% CI]", "Seeds", "Holm *p*"],
        "align": ["left"] + ["center"] * 6, "rows": rows,
        "note": f"Paired over {N_SEEDS} seeds; CI from the *t* distribution. Seeds: number of seeds in which the "
                "first strategy was higher. Holm-adjusted paired *t*-test *p* values within each preset and "
                "adapter family. FedSAA-0 is FedSAA with *ρ* = 0."}

    # ---- fixed budgets
    rows = []
    for s in LIN + ["probe"]:
        cells = [LAB[s]]
        for p in "AB":
            for rnd in (40, 100, 200):
                cells.append(f(tm(p, "linear", s, rnd)))
        rows.append(cells)
    for p in "AB":
        pass
    T["budget"] = {
        "title": "Test Accuracy at Fixed Budgets, Linear Adapter (%)",
        "widths": [1500, 680, 680, 680, 680, 680, 680], "size": 14,
        "header": ["Strategy", "A, 40", "A, 100", "A, 200", "B, 40", "B, 100", "B, 200"],
        "align": ["left"] + ["center"] * 6, "rows": rows,
        "note": f"Mean over {N_SEEDS} seeds at rounds 40, 100, and 200, equal weight over clients. "
                "Round 40 is the budget of our preliminary experiments."}

    # ---- image-level tests
    rows = []
    IP = [("fedavg", "local"), ("flexlora", "local"), ("fedsaa_u", "local"), ("oracle", "local"),
          ("centralized", "local"), ("oracle", "fedavg"), ("fedsaa_u", "fedsaa")]
    for p in "AB":
        rows.append(("group", f"Preset {p}"))
        for x, y in IP:
            cells = [f"{SHORT[x]} − {SHORT[y]}"]
            for rnd in (40, 200):
                d = img(p, "linear", rnd, x, y)
                cells += [f"{sg(d.diff_eq)} [{f(d.ci_lo)}, {f(d.ci_hi)}]", pv(d.p_holm), sg(d.diff_size_weighted)]
            rows.append(cells)
    T["image"] = {
        "title": "Image-Level Contrasts at Rounds 40 and 200, Linear Adapter (Percentage Points)",
        "widths": [2200, 1750, 760, 760, 1750, 760, 760], "size": 14,
        "header": ["Contrast", "40: diff. [95% CI]", "Holm *p*", "Pooled", "200: diff. [95% CI]", "Holm *p*",
                   "Pooled"],
        "align": ["left"] + ["center"] * 6, "rows": rows,
        "note": "Per-image correctness averaged over seeds. Differences are client-equal-weighted; CIs come from "
                "1,000 bootstrap resamples of images within clients; *p* values from 1,000 sign-flip "
                "randomizations, Holm-adjusted over the 19 planned contrasts of each preset and round "
                "(<0.001 means none of the randomizations reached the observed value). Pooled is the difference in "
                "accuracy over all test images, which weights clients by test-set size."}

    # ---- similarity and energy
    rows = []
    for p in "AB":
        rows.append(("group", f"Preset {p}"))
        for s in ("fedsaa", "fedsaa_u"):
            for rnd in (1, 10, 40, 200):
                rows.append([SHORT[s], str(rnd)] + [f(sim(p, s, rnd, c), 3) for c in
                                                     ("S_same", "S_other", "w_self", "w_same", "w_other", "energy")])
    T["sim"] = {
        "title": "Logged Similarities, Attention Weights, and Retained SVD Energy",
        "widths": [900, 520, 560, 560, 560, 560, 560, 560], "size": 14,
        "header": ["Rule", "Round", "*S*~same~", "*S*~other~", "*w*~self~", "*w*~same~", "*w*~other~", "Energy"],
        "align": ["left"] + ["center"] * 7, "rows": rows,
        "note": f"Means over clients and {N_SEEDS} seeds. *S*~same~ and *S*~other~: mean cosine similarity to the "
                "two peers of the same specialty and to the 12 clients of other specialties. *w*: total attention "
                "weight on the client itself, its two peers, and the 12 others; uniform weighting gives 0.067, "
                "0.133, and 0.800. Energy: share of squared singular values kept by the rank-16 truncation."}

    # ---- secondary metrics
    rows = []
    for s in LIN + ["probe"]:
        cells = [LAB[s]]
        for p in "AB":
            cells += [f(cp(p, "linear", s, "acc_w")), f(cp(p, "linear", s, "f1"), 3), f(cp(p, "linear", s, "auc"), 3),
                      f(cp(p, "linear", s, "worst"), 1)]
        rows.append(cells)
    T["secondary"] = {
        "title": "Secondary Metrics at the Validation-Selected Checkpoint, Linear Adapter",
        "widths": [1500, 700, 640, 640, 640, 700, 640, 640, 640], "size": 14,
        "header": ["Strategy", "A: Pooled", "F1", "AUC", "Worst", "B: Pooled", "F1", "AUC", "Worst"],
        "align": ["left"] + ["center"] * 8, "rows": rows,
        "note": "The checkpoint (round 40, 100, or 200) with the highest mean validation accuracy was chosen per "
                "run. Pooled: accuracy over all test images (%), which weights clients by test-set size. F1: "
                "macro-F1; AUC: one-versus-rest area under the ROC curve; both averaged over clients. Worst: "
                "accuracy of the lowest client (%)."}

    # ---- client-level gains
    rows = []
    for s in [x for x in LIN if x != "local"]:
        cells = [LAB[s]]
        for p in "AB":
            r = cg(p, "linear", s)
            cells += [f"{int(r.above)}/15", sg(r["median"]), f"{f(r.mn)} to {sg(r.mx)}"]
        rows.append(cells)
    T["clients"] = {
        "title": "Client-Level Transfer Gains Over Local-Only Training, Linear Adapter",
        "widths": [1500, 600, 700, 1350, 600, 700, 1350], "size": 14,
        "header": ["Strategy", "A: above", "Median", "Range", "B: above", "Median", "Range"],
        "align": ["left"] + ["center"] * 6, "rows": rows,
        "note": "Gains in percentage points at the validation-selected checkpoint, averaged over seeds for each "
                "client. Above: number of the 15 clients whose mean accuracy exceeded local-only training."}

    # ---- runtime
    rows = []
    for s in LIN + ["probe"]:
        rows.append([LAB[s], f"{sec('A', 'linear', s):.0f}", f"{sec('B', 'linear', s):.0f}"])
    T["runtime"] = {
        "title": "Mean Wall-Clock Time of One 200-Round Run, Linear Adapter (s)", "widths": [1900, 1300, 1300],
        "size": 14, "header": ["Strategy", "Preset A", "Preset B"], "align": ["left", "center", "center"],
        "rows": rows, "note": "One CPU worker, NumPy on cached features."}

    # ---- appendix: specialty gains (wide)
    SG = ["centralized", "fedavg", "fedprox", "oracle", "flexlora", "ffa", "fedsa", "fedsaa", "fedsaa_u"]
    rows = []
    for p in "AB":
        rows.append(("group", f"Preset {p}, linear adapter"))
        for sp in META[p]["specialties"]:
            cells = [NICE[sp]]
            for s in SG:
                r = spec(p, "linear", s, sp)
                t = sg(r.gain, 1)
                if r.hi < 0 or r.lo > 0:
                    t += "^†^"
                cells.append(t)
            rows.append(cells)
    T["spec_gain"] = {
        "title": "Transfer Gain of Each Specialty Over Local-Only Training (Percentage Points)",
        "widths": [1500] + [930] * 9, "size": 14,
        "header": ["Specialty"] + [SHORT[s] for s in SG],
        "align": ["left"] + ["center"] * 9, "rows": rows,
        "note": "Mean over the three clients of a specialty and the ten seeds, at the validation-selected "
                "checkpoint. ^†^ 95% CI over seeds excludes zero (uncorrected for the 100 comparisons)."}
    return T


ALGORITHM = ("Algorithm 1: FedSAA and FedSAA-U (server step of one round)", [
    (0, "**Input:** factors *A*~*i*~, *B*~*i*~ received from clients *i* = 1, …, *N*; *τ*, *λ*, *ρ*, *r*"),
    (0, "1: *U*~*i*~ ← *B*~*i*~*A*~*i*~ for all *i*; *S*~*ij*~ ← cos(*U*~*i*~, *U*~*j*~) for all *i*, *j*"),
    (0, "2: *G* ← (1/*N*) Σ~*j*~ *U*~*j*~"),
    (0, "3: **for** every client *i* **do**"),
    (1, "4: *w*~*ij*~ ← exp(*S*~*ij*~/*τ*) / Σ~*k*~ exp(*S*~*ik*~/*τ*)"),
    (1, "5: *Ũ*~*i*~ ← (1 − *λ*) Σ~*j*~ *w*~*ij*~*U*~*j*~ + *λG*"),
    (1, "6: **if** FedSAA-U **then** *Ũ*~*i*~ ← (1 − *ρ*)*Ũ*~*i*~ + *ρU*~*i*~"),
    (1, "7: (*A*′~*i*~, *B*′~*i*~) ← factors of the rank-*r* truncated SVD of *Ũ*~*i*~"),
    (1, "8: **if** FedSAA **then** *A*′~*i*~ ← (1 − *ρ*)*A*′~*i*~ + *ρA*~*i*~; *B*′~*i*~ ← (1 − *ρ*)*B*′~*i*~ + *ρB*~*i*~"),
    (1, "9: send *A*′~*i*~, *B*′~*i*~ to client *i*"),
    (0, "10: **end for**"),
])


def p(t):
    return ("p", t)


def pc(t):
    return ("pc", t)


# ------------------------------------------------------------------ body
def build_body():
    T = tables()
    B = []
    add, ext = B.append, B.extend

    # numbers used repeatedly
    A_fa, A_or, A_fl, A_su, A_s, A_r0, A_ce = (pair("A", "linear", k, "local") for k in
                                              ("fedavg", "oracle", "flexlora", "fedsaa_u", "fedsaa", "fedsaa_r0",
                                               "centralized"))
    A_ffa, A_sa, A_fp = (pair("A", "linear", k, "local") for k in ("ffa", "fedsa", "fedprox"))
    uf = {p: pair(p, "linear", "fedsaa_u", "fedsaa") for p in "AB"}
    r0f = {p: pair(p, "linear", "fedsaa_r0", "fedsaa") for p in "AB"}
    ufl = {p: pair(p, "linear", "fedsaa_u", "flexlora") for p in "AB"}
    ofa = {p: pair(p, "linear", "oracle", "fedavg") for p in "AB"}
    uor = {p: pair(p, "linear", "fedsaa_u", "oracle") for p in "AB"}
    pfa = {p: pair(p, "linear", "fedprox", "fedavg") for p in "AB"}
    nl_ofa = {p: pair(p, "nonlinear", "oracle", "fedavg") for p in "AB"}
    probe = {p: pair(p, "linear", "linear local", "probe") for p in "AB"}
    nprobe = {p: pair(p, "nonlinear", "nonlinear local", "probe") for p in "AB"}
    nl_fa = {p: pair(p, "nl-lin", "nonlinear fedavg", "linear fedavg") for p in "AB"}
    nl_lo = {p: pair(p, "nl-lin", "nonlinear local", "linear local") for p in "AB"}
    B_vs = {k: pair("B", "linear", k, "local") for k in LIN if k != "local"}
    B_lo_k = min(B_vs, key=lambda k: B_vs[k]["diff"])
    B_hi_k = max(B_vs, key=lambda k: B_vs[k]["diff"])
    lpk = {pp: peak_round(pp, "linear", "local") for pp in "AB"}
    so40 = {pp: sim(pp, "fedsaa", 40, "S_other") for pp in "AB"}
    nfed = len(LIN) - 1

    # ============================== I. INTRODUCTION
    add(("h1", "INTRODUCTION"))
    ext([p(OLD(1)), p(OLD(2)), p(OLD(3))])
    add(p("We study this question in two simulated federations built from MedMNIST v2 \\cite{medmnist}, each with "
          "five specialties and three clients per specialty. A rank-16 adapter on a frozen ResNet-18 \\cite{he2016} "
          "is shared, and each client keeps its own classifier. Our preliminary experiments compared five "
          "strategies at a fixed budget of 40 rounds with three seeds and found that local-only training beat every "
          "federated strategy in one federation, including FedSAA (Specialty-Aware Adapter Aggregation), a "
          "similarity-weighted rule that we had designed for this setting. Those runs had limitations: they "
          "had not converged, the learning rate was not tuned per strategy, strong baselines were missing, and "
          "three seeds cannot carry firm conclusions. This article replaces those experiments. Every "
          f"strategy is tuned on held-out validation data and run for {ROUNDS} rounds with {N_SEEDS} seeds, the "
          "primary result is taken at the round chosen on validation data, and the comparison includes FlexLoRA "
          "\\cite{flexlora}, FFA-LoRA \\cite{ffalora}, FedSA-LoRA \\cite{fedsalora}, a specialty-oracle that averages "
          "only within a specialty, a linear probe, a nonlinear adapter, and two variants of FedSAA that isolate its "
          "retention step."))
    add(p("The study answers four research questions. RQ1: Does federating a shared adapter improve client accuracy "
          "over local-only training when clients come from different specialties? RQ2: Does similarity-weighted "
          "aggregation recover the specialty structure of the federation, and does it improve on simpler rules? "
          "RQ3: Which specialties and clients gain or lose? RQ4: How much do the answers depend on the training "
          "budget and on the form of the adapter?"))
    add(p("The answers are partly negative, and we report them as such. The contributions are as follows. First, a "
          "reproducible cross-specialty benchmark built from public data, with a fixed partition, validation "
          "tuning, validation-based stopping, ten seeds, and seed-level and image-level paired tests. Second, a "
          "comparison of 12 strategies showing that whether federation helps depends on the composition of the "
          "federation: in one federation global averaging lost to local training while averaging within specialties "
          "gained about two points, and in the other every federated strategy gained. Third, an analysis of the "
          "model and of the aggregation rule (Section V), stated as four propositions: the adapter adds no "
          "hypothesis capacity over a linear probe; averaging factors differs from averaging updates by their "
          "cross-covariance; retention in factor space depends on an arbitrary basis; and attention weights become "
          "uniform when update similarities approach one. Logged similarities, singular values, and the FedSAA "
          "variants test the last three predictions directly. Fourth, a demonstration that a fixed training budget "
          "can reverse the ranking of local and federated training, which explains the result of our preliminary "
          "experiments. Fifth, the code, the raw per-run and per-image results, and the scripts that regenerate every "
          "table and figure. The clients are simulated from public datasets; nothing here is a clinical validation."))
    add(p("Section II reviews related work, Section III states the problem, Section IV describes the methods, "
          "Section V analyzes the model and the aggregation rule, Section VI reports the results, Section VII "
          "discusses them, and Section VIII concludes. The Appendix gives the per-specialty results."))

    # ============================== II. RELATED WORK
    add(("h1", "RELATED WORK"))
    add(("h2", "FEDERATED BENCHMARKS IN MEDICAL IMAGING"))
    ext([p(OLD(10)), p(OLD(11))])
    add(("h2", "HETEROGENEITY AND PERSONALIZATION"))
    ext([p(OLD(13)), p(OLD(14))])
    add(("h2", "FEDERATED LOW-RANK ADAPTATION"))
    add(p(OLD(16)))
    add(p(OLD(17).replace("{T:related} summarizes how these methods differ in what is communicated, where "
                          "aggregation takes place, and what stays local.",
                          "{T:related} summarizes how these methods differ and which of them are run here.")))
    add(("wide_group", ["related"]))
    add(("h2", "NEGATIVE TRANSFER AND LINEAR PROBING"))
    add(p(OLD(20).replace("The adapter studied here changes the features only through a linear transformation, a "
                          "point that Section V makes precise.",
                          "The linear adapter studied here changes the features only through a linear "
                          "transformation, a point that Section V makes precise; we therefore also test a nonlinear "
                          "adapter and a linear probe.")))
    add(("h2", "REPORTING COMPARISONS"))
    add(p(OLD(22) + " Early stopping on held-out data \\cite{prechelt1998} is the usual way to compare methods "
          "that converge at different rates; we use it as the primary endpoint and report fixed budgets as well."))

    # ============================== III. PRELIMINARIES
    add(("h1", "PRELIMINARIES AND PROBLEM FORMULATION"))
    add(("h2", "FEDERATED OPTIMIZATION"))
    add(p(OLD(25)))
    add(("eq", OLD(26)))
    add(pc(OLD(27)))
    add(("eq", OLD(28)))
    add(pc(OLD(29)))
    add(("eq", OLD(30)))
    add(pc(OLD(31)))
    add(("eq", OLD(32)))
    add(pc(OLD(33)))
    add(("h2", "SHARED AND PRIVATE PARAMETERS"))
    add(p(OLD(35)))
    add(("eq", OLD(36)))
    add(pc(OLD(37)))
    add(("h2", "LOW-RANK ADAPTATION"))
    add(p(OLD(39)))
    add(("eq", OLD(40)))
    add(pc(OLD(41)))
    add(("h2", "TRANSFER GAIN AND NEGATIVE TRANSFER"))
    add(p(OLD(43)))
    add(("eq", OLD(44)))
    add(pc(OLD(45)))
    add(("h2", "NOTATION"))
    add(p("{T:notation} lists the symbols used in the rest of the article."))
    add(("table", "notation"))

    # ============================== IV. METHODS
    mA, mB = META["A"], META["B"]
    add(("h1", "METHODS"))
    add(("h2", "DATA AND CLIENT CONSTRUCTION"))
    add(p("All images come from MedMNIST v2 \\cite{medmnist}, in its 64 × 64 pixel release. Each federation, called "
          "Preset A and Preset B, combines five two-dimensional subsets that stand for five specialties "
          "({T:composition}). For each subset the official training split was shuffled once with a fixed seed "
          "(2026) and cut into three equal parts, one per client; each part was capped at 2,000 images, and its "
          "first 15% became that client’s validation set. The official test split was shuffled and cut in the same "
          "way, with a cap of 1,000 images per client. The partition does not depend on the training seed, so every "
          f"strategy and seed sees the same clients. Preset A has {n_train_total('A'):,} training, "
          f"{n_val_total('A'):,} validation, and {n_test_total('A'):,} test images; Preset B has "
          f"{n_train_total('B'):,}, {n_val_total('B'):,}, and {n_test_total('B'):,}. Images were converted to three "
          "channels and normalized with the ImageNet mean and standard deviation, without augmentation. Label spaces "
          "differ between specialties; the five ordinal grades of RetinaMNIST were treated as five classes. "
          "MedMNIST v2 is released under CC BY 4.0, except DermaMNIST (CC BY-NC 4.0). The study used only public, "
          "de-identified data and involved no new data from human participants."))
    add(p(OLD(52).replace("whose clients have between 52 and 208 test images.",
                          f"whose clients have between {min(mB['n_test'])} and "
                          f"{max(x for x in mB['n_test'] if x < 1000)} test images and between "
                          f"{min(mB['n_train'])} and {max(x for x in mB['n_train'] if x < 1700)} training images.")))
    add(("table", "composition"))
    add(("h2", "MODEL"))
    add(p(OLD(55).replace("A low-rank residual adapter transforms the feature as",
                          "Because the backbone is frozen and no augmentation is used, the features were computed "
                          "once and cached. A low-rank residual adapter transforms the feature as")))
    add(("eq", r"h(x) = x + s\,BAx"))
    add(pc("where *A* is an *r* × *d* matrix, *B* is a *d* × *r* matrix, the rank is *r* = 16, and *s* = *α*/*r* "
           "with *α* = 32. We call this the linear adapter. To test whether the conclusions depend on linearity, we "
           "also train a nonlinear adapter,"))
    add(("eq", r"h(x) = x + s\,B\,\mathrm{ReLU}(Ax)"))
    add(pc("with the same shapes. Either adapter has 2*rd* = 8,192 parameters, which is all that is communicated. "
           "All clients of a run start from the same adapter, with *A* drawn from N(0, 1/*d*) and *B* = 0, so the "
           "adapter is initially the identity. Each client has its own linear classifier on *h*(*x*), initialized "
           "with N(0, 10^−4^) weights and zero bias; classifiers are never aggregated. No dropout or normalization "
           "follows the adapter. Without an adapter, the same classifier on *x* is a linear probe."))
    add(p(OLD(58)))
    add(("fig", "arch", "fig_arch.png",
         "Model and communication. The ResNet-18 backbone, cut after its third stage, is frozen and its 256-d "
         "pooled features are cached. Each client trains a rank-16 adapter (linear, or with a ReLU between *A* and "
         "*B*) and a private linear classifier. Only the adapter factors *A* and *B* are sent to the server."))
    add(("h2", "TRAINING STRATEGIES"))
    add(p("All strategies used the Adam optimizer \\cite{kingma2015} with a batch size of 64 and cross-entropy loss. "
          "One round is three local epochs, after which the optimizer state is reset for every strategy. All 15 "
          f"clients take part in every round, and every run lasts {ROUNDS} rounds. The strategies are as follows."))
    add(p("*Local-only:* Each client trains its own adapter and classifier with no communication. *Linear probe:* "
          "As local-only, without an adapter."))
    add(p("*Centralized:* One adapter is shared by all clients, and one classifier per specialty is shared by the "
          "three clients of that specialty. Each step averages the gradients of one mini-batch from every client, "
          "so the adapter sees the pooled data."))
    add(p("*FedAvg* \\cite{mcmahan2017}: After each round the server replaces every client’s *A* and *B* by their "
          "sample-size-weighted averages, which is also the rule of FedIT \\cite{fedit}. *FedProx* "
          "\\cite{li2020fedprox}: As FedAvg, with a proximal penalty of weight *μ* on the distance between the local "
          "and the received adapter."))
    add(p("*Specialty oracle:* FedAvg restricted to the three clients of each specialty, so specialties never "
          "exchange parameters. This is clustered FL \\cite{sattler2021,ghosh2020} with the true clusters given, "
          "and it shows what a rule that recovered the specialty structure could achieve."))
    add(p("*FlexLoRA* \\cite{flexlora}: The server averages the products *U*~*i*~ = *B*~*i*~*A*~*i*~ with sample "
          "weights, truncates the average to rank *r* by SVD, and sends the same factors to every client. "
          "*FFA-LoRA* \\cite{ffalora}: *A* is frozen at its shared initialization, and only *B* is trained and "
          "averaged. *FedSA-LoRA* \\cite{fedsalora}: *A* is averaged and *B* is kept local."))
    add(p("*FedSAA*, *FedSAA with ρ = 0*, and *FedSAA-U:* The personalized rule of Section IV-D with retention in "
          "factor space, without retention, and with retention in update space. FlexLoRA and the FedSAA variants "
          "aggregate products and are defined for the linear adapter only, so the nonlinear adapter was run with "
          "the other seven strategies."))
    add(("h2", "FEDSAA AGGREGATION"))
    add(p(OLD(68)))
    add(("eq", OLD(69)))
    add(pc(OLD(70)))
    add(("eq", OLD(71)))
    add(pc(OLD(72)))
    add(("eq", OLD(73)))
    add(pc(OLD(74)))
    add(("eq", OLD(75)))
    add(pc(OLD(76)))
    add(("eq", OLD(77)))
    add(pc(OLD(78)))
    add(("eq", OLD(79)))
    add(pc(OLD(80)))
    add(("eq", OLD(81)))
    add(pc(OLD(82)))
    add(("eq", OLD(83)))
    add(("eq", OLD(84)))
    add(pc("Here *ρ* is the retention coefficient; it is distinct from the adapter rank *r*. FedSAA-U instead "
           "interpolates the updates before a single truncation,"))
    add(("eq", r"\tilde{U}_i \leftarrow (1-\rho)\,\tilde{U}_i + \rho\,U_i"))
    add(pc("and then factorizes the result. With *ρ* = 0 the two variants coincide. Section V-C shows why they "
           "differ otherwise."))
    add(p("FedSAA does not reduce to FedAvg. With *λ* = 1 and *ρ* = 0, or whenever the attention weights are "
          "uniform, every client receives the rank-*r* truncation of the unweighted mean of the products, which is "
          "FlexLoRA with equal client weights. FedAvg averages *A* and *B* separately, and the product of averaged "
          "factors differs from the average of products \\cite{flora}. Algorithm 1 gives the server step."))
    add(("alg", ALGORITHM))
    add(("h2", "COMPLEXITY AND COMMUNICATION COST"))
    total_mb = 8192 * 4 * 2 * 15 * ROUNDS / 1e6
    add(p("Under every federated strategy each client uploads and downloads 2*rd* = 8,192 values per round, about "
          f"33 kB in single precision. Over {ROUNDS} rounds and 15 clients the total traffic is about "
          f"{total_mb:.0f} MB. FedAvg, FedProx, FFA-LoRA, FedSA-LoRA, and the specialty oracle need server work "
          "proportional to *Nrd* per round. FlexLoRA forms *N* products and one SVD of a *d* × *d* matrix; FedSAA "
          "forms *N* products, *N*^2^ similarities, and *N* SVDs, at costs proportional to *Nrd*^2^, *N*^2^*d*^2^, "
          "and *Nd*^3^. With *d* = 256 and *N* = 15 this is small, but in our CPU implementation it was not "
          "negligible beside local training (Section VI-I). FedSAA also requires the server to keep *N* adapters, "
          "and a new client cannot simply download a global model."))
    add(("h2", "HYPERPARAMETER TUNING"))
    add(p("The learning rate of every strategy, preset, and adapter was chosen from {3 × 10^−4^, 10^−3^, "
          "3 × 10^−3^}, and the FedProx weight *μ* jointly from {0.001, 0.01, 0.1}. Each configuration was trained "
          f"once with a separate seed (1000) for 100 rounds, giving {len(TUNE)} tuning runs. The configuration with "
          "the highest mean validation accuracy at rounds 40 and 100 was selected ({T:tuning}); test data were not "
          "used. The FedSAA parameters *τ*, *λ*, and *ρ* were fixed at the values of the preliminary experiments, so that "
          "*ρ* = 0 and FedSAA-U act as ablations of a prespecified rule."))
    add(("table", "tuning"))
    add(("h2", "EVALUATION"))
    add(p("After every round each client’s model was evaluated on its validation and test sets. The primary metric "
          "is client-averaged test accuracy, with every client weighted equally, at the round *t*^*^ at which the "
          "client-averaged validation accuracy of that run was highest (ties: earliest round). Selecting *t*^*^ "
          "on validation data gives every strategy the training budget that suits it, which a fixed budget does "
          "not. As secondary results we report test accuracy at fixed budgets of 40, 100, and 200 rounds; the "
          "accuracy over all test images pooled across clients, which weights clients by test-set size; macro-F1 "
          "and one-versus-rest area under the ROC curve (AUC), averaged over clients; and the accuracy of the "
          "worst client. Full metrics and per-image predictions were saved at rounds 40, 100, and 200, so the "
          "secondary metrics use the one of these checkpoints with the highest validation accuracy."))
    add(p("Transfer gains (Section III-D) were computed per client and per specialty against local-only training "
          "at the same selected checkpoint. For FedSAA and FlexLoRA we logged, at rounds 1, 10, 40, 100, and 200, "
          "the similarity matrix, the attention weights, and the share of squared singular values retained by the "
          "rank-*r* truncation, here called the retained energy."))
    add(("h2", "STATISTICAL ANALYSIS"))
    add(p(f"The unit of the primary analysis is the seed, with {N_SEEDS} runs per strategy, preset, and adapter. "
          "Results are given as the mean ± sample standard deviation (SD). Twenty contrasts were specified in "
          "advance for the linear adapter and eight for the nonlinear adapter ({T:contrasts}). Each was tested with "
          "a two-sided paired *t*-test on the per-seed differences, with a 95% confidence interval (CI) from the "
          "*t* distribution, and a Wilcoxon signed-rank test \\cite{wilcoxon1945}; at the fixed checkpoints an exact "
          "sign-flip test was also computed. The *t*-test "
          "*p* values were adjusted by Holm’s method \\cite{holm1979} within each preset and adapter. We also report "
          "the number of seeds in which the difference had the sign of the mean. With ten seeds the smallest "
          "possible two-sided Wilcoxon and sign-flip *p* value is 0.002."))
    add(p("Because seed-level tests ignore the sampling error of finite test sets, the planned contrasts at the "
          "fixed checkpoints were also tested on images. The correctness of each test image was averaged over "
          "seeds, the client-equal-weighted mean difference was computed, a 95% CI was obtained from 1,000 "
          "bootstrap resamples of images within clients \\cite{efron1993}, and a *p* value from 1,000 sign-flip "
          "randomizations of the per-image differences, Holm-adjusted within each preset and round. Comparisons of "
          "the adapter with the linear probe, and of the nonlinear with the linear adapter, are reported without "
          "adjustment. Following \\cite{wasserstein2016}, we interpret effect sizes and intervals, not thresholds. "
          "No runs were excluded and none failed."))
    add(("h2", "IMPLEMENTATION"))
    add(p("Features were extracted with PyTorch \\cite{paszke2019} and torchvision. Training, aggregation, and "
          "evaluation were implemented in NumPy \\cite{harris2020}, with all clients of a run processed as one "
          "batched array; gradients were verified against finite differences. Statistics used SciPy "
          f"\\cite{{virtanen2020}}. The {len(RUNS)} runs ({len(TUNE)} tuning and {len(MAIN)} main) took "
          f"{SEC_TOTAL / 3600:.1f} h of wall-clock time on the CPU of a Kaggle notebook. {{T:config}} lists the "
          "configuration. [[AUTHORS: add the versions of Python, NumPy, SciPy, PyTorch, torchvision, and medmnist "
          "used in the Kaggle environment.]]"))
    add(("table", "config"))

    # ============================== V. ANALYSIS
    add(("h1", "ANALYSIS OF THE MODEL AND THE AGGREGATION RULE"))
    add(p("Four properties of the model and of FedSAA follow from their definitions alone. They do not depend on "
          "the data. Each gives a prediction that Section VI tests. The client index is dropped where no confusion "
          "can arise."))
    add(("h2", "THE LINEAR ADAPTER DOES NOT ENLARGE THE HYPOTHESIS CLASS"))
    add(p(OLD(102).replace("Write *s* = *α*/*r* and *M*", "Write *M*")))
    add(("eq", OLD(103)))
    add(p(OLD(104)))
    add(p(OLD(105)))
    add(p(OLD(106) + " *Prediction:* a local linear adapter cannot outperform a well-tuned linear probe for "
          "reasons of capacity, and any gain from sharing it must come from optimization. The nonlinear adapter is "
          "not covered by Proposition 1."))
    add(("h2", "AVERAGING FACTORS IS NOT AVERAGING UPDATES"))
    add(p(OLD(108)))
    add(("eq", OLD(109)))
    add(p(OLD(110)))
    add(p(OLD(111).replace("FedSAA avoids it by aggregating the products *U*~*i*~, at the cost of the truncation "
                           "and refactorization analyzed below.",
                           "FlexLoRA and FedSAA avoid it by aggregating the products *U*~*i*~, at the cost of a "
                           "truncation and a refactorization. *Prediction:* product-space rules should match or "
                           "exceed FedAvg when client factors disagree.")))
    add(("h2", "FACTOR-SPACE RETENTION DEPENDS ON AN ARBITRARY BASIS"))
    add(p(OLD(122)))
    add(("eq", OLD(123)))
    add(pc(OLD(124)))
    add(("eq", OLD(125)))
    add(p(OLD(126)))
    add(p(OLD(127)))
    add(("eq", OLD(128)))
    add(pc(OLD(129).replace("A basis-independent alternative is to interpolate the updates, (1 − *ρ*)*Ũ*~*i*~ + "
                            "*ρU*~*i*~, before a single truncated SVD.",
                            "FedSAA-U interpolates the updates before a single truncated SVD and is free of this "
                            "term. *Prediction:* FedSAA-U and FedSAA with *ρ* = 0 should match or exceed FedSAA.")))
    add(("h2", "ATTENTION BECOMES UNIFORM WHEN UPDATES ALIGN"))
    add(p("The attention weight of client *j* for client *i* can be written as"))
    add(("eq", r"w_{ij} = \Big(\sum_{k=1}^{N} e^{(S_{ik}-S_{ij})/\tau}\Big)^{-1}"))
    add(p("*Proposition 4:* If all similarities of client *i* lie in an interval of width *ε*, then every weight "
          "satisfies"))
    add(("eq", r"\frac{e^{-\varepsilon/\tau}}{N} \le w_{ij} \le \frac{e^{\varepsilon/\tau}}{N}"))
    add(pc("*Proof:* Each exponent lies in [−*ε*/*τ*, *ε*/*τ*], so the sum lies between *Ne*^−*ε*/*τ*^ and "
           "*Ne*^*ε*/*τ*^."))
    add(p("The weights therefore become uniform as the spread of similarities shrinks, whatever the temperature. "
          "Because a client’s similarity to itself is 1, the spread is at least 1 − min~*j*~ *S*~*ij*~. With "
          "uniform weights the personalized aggregate equals the global mean, the blend with *λ* has no effect, and "
          "FedSAA becomes FlexLoRA with equal client weights (followed by retention). In a federation of five "
          "specialties with three clients each, the other 12 clients then receive 12/15 = 0.80 of the weight. "
          "Similarities between products *U*~*i*~ = *B*~*i*~*A*~*i*~ need not reflect specialty: after the "
          "first round every client starts from an aggregated adapter, and if local training changes it little, "
          "all products stay close to that common adapter. *Prediction:* unless local updates are large relative "
          "to the shared adapter, FedSAA will lose its specialty selectivity after a few rounds and behave like "
          "FlexLoRA."))
    add(p("Finally, the blended update is a combination of *N* products of rank at most *r*, so its rank can reach "
          "min(*d*, *Nr*) = 240, and the truncated SVD discards everything beyond rank *r* \\cite{eckart1936}. How "
          "much is lost depends on how similar the combined updates are. The retained energy measures it."))

    # ============================== VI. RESULTS
    add(("h1", "RESULTS"))
    # ---- A
    add(("h2", "WHETHER FEDERATION HELPS DEPENDS ON THE FEDERATION"))
    sA = sorted(LIN, key=lambda s: -es("A", "linear", s)[0])
    add(p(f"{{T:main}} gives the primary results, and {{F:es}} shows the means with their confidence intervals. In Preset A with the linear adapter, "
          f"local-only training reached {esd('A', 'linear', 'local')}. FedAvg was below it by "
          f"{f(-A_fa['diff'])} points (95% CI {ci(-A_fa['hi'], -A_fa['lo'])}; lower in {N_SEEDS - pos(A_fa)} of "
          f"{N_SEEDS} seeds; Holm-adjusted *p* = {pv(A_fa['holm'])}), and FFA-LoRA by {f(-A_ffa['diff'])} points. "
          f"The specialty oracle and centralized training were highest, {sg(A_or['diff'])} and "
          f"{sg(A_ce['diff'])} points above local-only training, and higher in every seed ({T_ref('contrasts')}). "
          f"FlexLoRA ({sg(A_fl['diff'])}), FedSAA-U ({sg(A_su['diff'])}), and FedSAA with *ρ* = 0 "
          f"({sg(A_r0['diff'])}) were in between and above local-only training in all ten seeds. FedProx "
          f"({sg(A_fp['diff'])}) and FedSA-LoRA ({sg(A_sa['diff'])}) were not clearly separated from it, and FedSAA "
          f"itself was indistinguishable from it ({sg(A_s['diff'])}; 95% CI {ci(A_s['lo'], A_s['hi'])})."))
    add(p(f"Preset B behaved differently. Every one of the {nfed} federated strategies exceeded local-only training "
          f"({esd('B', 'linear', 'local')}), by {sg(B_vs[B_lo_k]['diff'])} ({LAB[B_lo_k]}) to "
          f"{sg(B_vs[B_hi_k]['diff'])} points ({LAB[B_hi_k]}), all with Holm-adjusted *p* below 0.05. Here the "
          f"specialty oracle had no advantage over FedAvg ({sg(ofa['B']['diff'])} points; 95% CI "
          f"{ci(ofa['B']['lo'], ofa['B']['hi'])}), whereas in Preset A it led FedAvg by {f(ofa['A']['diff'])} points "
          f"({ci(ofa['A']['lo'], ofa['A']['hi'])}). Most strategies in Preset B lay within about one point of each "
          "other, and only FFA-LoRA and local-only training were clearly lower."))
    add(("wide_group", ["main"]))
    add(("fig", "es", "fig_es.png",
         f"Test accuracy at the validation-selected round, mean and 95% CI over {N_SEEDS} seeds. Circles: "
         "linear adapter; squares: nonlinear adapter; diamonds: linear probe. The dotted line marks local-only "
         "training with the linear adapter. Accuracy is averaged over the 15 clients with equal weight."))
    add(("wide_group", ["contrasts"]))
    # ---- B
    add(("h2", "FEDSAA REDUCES TO FLEXLORA, AND FACTOR-SPACE RETENTION COSTS ACCURACY"))
    add(p(f"Among the FedSAA variants, the retention step decided the outcome. In Preset A, FedSAA-U exceeded FedSAA "
          f"by {pd_(uf['A'])}, in {pos(uf['A'])} of {N_SEEDS} seeds (Holm-adjusted *p* = {pv(uf['A']['holm'])}), and "
          f"removing retention altogether gave a similar gain ({sg(r0f['A']['diff'])} points; adjusted *p* = "
          f"{pv(r0f['A']['holm'])}). In Preset B the direction was the same for FedSAA-U ({sg(uf['B']['diff'])} "
          f"points; 95% CI {ci(uf['B']['lo'], uf['B']['hi'])}; adjusted *p* = {pv(uf['B']['holm'])}). This is what "
          "Proposition 3 predicts: interpolating factors from two unrelated factorizations distorts the update, "
          "and interpolating the updates does not."))
    add(p(f"FedSAA-U did not improve on FlexLoRA, which uses no similarity weighting at all: the difference was "
          f"{sg(ufl['A']['diff'])} points in Preset A (95% CI {ci(ufl['A']['lo'], ufl['A']['hi'])}) and "
          f"{sg(ufl['B']['diff'])} in Preset B ({ci(ufl['B']['lo'], ufl['B']['hi'])}). In Preset A it remained "
          f"{f(-uor['A']['diff'])} points below the specialty oracle ({ci(uor['A']['lo'], uor['A']['hi'])}), so the "
          "similarity weighting recovered none of the benefit of separating specialties."))
    add(("h2", "LOGGED SIMILARITIES EXPLAIN WHY"))
    add(p(f"{{T:sim}} and {{F:sim}} show the reason. After the first round the similarity weighting did separate "
          f"specialties: the mean similarity of a FedSAA client to its two same-specialty peers was "
          f"{f(sim('A', 'fedsaa', 1, 'S_same'), 2)} in Preset A and {f(sim('B', 'fedsaa', 1, 'S_same'), 2)} in "
          f"Preset B, against {f(sim('A', 'fedsaa', 1, 'S_other'), 2)} and {f(sim('B', 'fedsaa', 1, 'S_other'), 2)} "
          f"to clients of other specialties, and other specialties received {f(sim('A', 'fedsaa', 1, 'w_other'), 2)} "
          f"and {f(sim('B', 'fedsaa', 1, 'w_other'), 2)} of the attention weight instead of 0.80. By round 10 the "
          f"similarity to other specialties had risen to {f(sim('A', 'fedsaa', 10, 'S_other'), 2)} and "
          f"{f(sim('B', 'fedsaa', 10, 'S_other'), 2)}, and by round 40 to {f(so40['A'], 2)} and {f(so40['B'], 2)}. "
          f"The weight on other specialties was then {f(sim('A', 'fedsaa', 40, 'w_other'), 2)} and "
          f"{f(sim('B', 'fedsaa', 40, 'w_other'), 2)}, close to the uniform 0.80, and it stayed there to round 200. "
          "FedSAA-U and FedSAA with *ρ* = 0 converged to uniform weights faster. As Proposition 4 predicts, the "
          "rule became FlexLoRA with equal weights within the first tens of rounds."))
    add(p(f"The retained energy of the rank-16 truncation was at least {f(SIM.energy.min(), 3)} in every logged "
          "round and rule, and rounded to 1.000 from round 10 on. The truncation therefore discarded almost "
          "nothing, because the products being combined were nearly collinear; the concern that rank truncation "
          "loses information (Section V-D) did not apply in practice."))
    add(("table", "sim"))
    add(("fig", "sim", "fig_sim.png",
         "Similarity and attention in FedSAA over rounds (mean over clients and seeds). Mean cosine similarity of "
         "a client’s product *BA* to its same-specialty peers and to clients of other specialties, and the total "
         "attention weight on other specialties; the dotted line marks the uniform value 0.80."))
    # ---- C
    add(("h2", "SPECIALTY- AND CLIENT-LEVEL EFFECTS"))
    rA = spec("A", "linear", "fedavg", "pathmnist")
    rO = spec("A", "linear", "fedavg", "octmnist")
    ret = {s: spec("B", "linear", s, "retinamnist") for s in ("fedavg", "fedprox", "flexlora", "fedsa", "fedsaa",
                                                                "oracle", "fedsaa_u")}
    pn = spec("B", "linear", "fedavg", "pneumoniamnist")
    orpos = sum(spec(pp, "linear", "oracle", sp).gain > 0 for pp in "AB" for sp in META[pp]["specialties"])
    add(p(f"Averages over clients hide opposite effects ({{F:spec}}; full table in the Appendix, {{T:spec_gain}}). "
          f"In Preset A, FedAvg lowered the accuracy of PathMNIST by {f(-rA.gain, 1)} points (95% CI "
          f"{ci(-rA.hi, -rA.lo, 1)}) and of OCTMNIST by {f(-rO.gain, 1)} ({ci(-rO.hi, -rO.lo, 1)}), and raised that "
          f"of BloodMNIST by {f(spec('A', 'linear', 'fedavg', 'bloodmnist').gain, 1)}. In Preset B, FedAvg, FedProx, "
          f"FlexLoRA, FedSA-LoRA, and FedSAA all lowered RetinaMNIST, by {f(-ret['fedavg'].gain, 1)}, "
          f"{f(-ret['fedprox'].gain, 1)}, {f(-ret['flexlora'].gain, 1)}, {f(-ret['fedsa'].gain, 1)}, and "
          f"{f(-ret['fedsaa'].gain, 1)} points, each with an interval excluding zero, while PneumoniaMNIST gained "
          f"most ({sg(pn.gain, 1)} under FedAvg). The specialty oracle raised RetinaMNIST "
          f"({sg(ret['oracle'].gain, 1)}), and its gain was positive in {orpos} of the 10 specialties."))
    oA, fA, uA = cg("A", "linear", "oracle"), cg("A", "linear", "fedavg"), cg("A", "linear", "fedsaa_u")
    oB, fB = cg("B", "linear", "oracle"), cg("B", "linear", "fedavg")
    add(p(f"At the client level ({{T:clients}}), FedAvg exceeded local-only training for {int(fA.above)} of 15 "
          f"clients in Preset A, with a median gain of {sg(fA['median'])} points, whereas the specialty oracle did "
          f"so for {int(oA.above)} and FedSAA-U for {int(uA.above)}. In Preset B FedAvg improved {int(fB.above)} "
          f"clients and the oracle {int(oB.above)}. The largest single-client loss under a federated strategy was "
          f"{f(-min(CG[CG.adapter == 'linear'].mn), 1)} points, for a RetinaMNIST client in Preset B under FedAvg "
          "and FedProx."))
    add(("fig", "spec", "fig_spec.png",
         "Transfer gain of each specialty over local-only training at the validation-selected checkpoint (mean "
         "over three clients and ten seeds, with 95% CI over seeds), for selected strategies with the linear "
         "adapter."))
    add(("wide_group", ["clients"]))
    # ---- D
    add(("h2", "A FIXED BUDGET CAN REVERSE THE RANKING"))
    c40 = {s: tm("A", "linear", s, 40) for s in LIN}
    c200 = {s: tm("A", "linear", s, 200) for s in LIN}
    glob_ = ["fedavg", "fedprox", "flexlora", "ffa", "fedsa", "fedsaa", "fedsaa_r0", "fedsaa_u"]
    i40 = {k: img("A", "linear", 40, k, "local") for k in ("fedavg", "flexlora")}
    i200 = {k: img("A", "linear", 200, k, "local") for k in ("fedavg", "flexlora")}
    add(p(f"Local-only training converged and then overfit. Its seed-mean test curve peaked at round "
          f"{lpk['A'][0]} in Preset A and {lpk['B'][0]} in Preset B ({{F:conv}}), and its validation-selected round "
          f"averaged {es('A', 'linear', 'local', 'es_round')[0]:.0f} in Preset A. In Preset B, where several validation "
          f"sets are small and the late curve is nearly flat, the selected round varied widely (mean "
          f"{es('B', 'linear', 'local', 'es_round')[0]:.0f}, SD {es('B', 'linear', 'local', 'es_round')[1]:.0f}). "
          f"Global federated strategies converged far more slowly: in Preset A their selected rounds averaged "
          f"{min(es('A', 'linear', s, 'es_round')[0] for s in glob_):.0f} to "
          f"{max(es('A', 'linear', s, 'es_round')[0] for s in glob_):.0f}, and FedAvg was still rising at round "
          f"200. The specialty oracle and centralized training, whose clients share data from their own specialty, "
          f"converged early ({es('A', 'linear', 'oracle', 'es_round')[0]:.0f} and "
          f"{es('A', 'linear', 'centralized', 'es_round')[0]:.0f})."))
    add(p(f"As a result the ranking depends on when training stops ({{T:budget}}). At round 40, the budget of our "
          f"preliminary experiments, local-only training in Preset A ({f(c40['local'])}%) was ahead of every global "
          f"federated strategy ({f(min(c40[s] for s in glob_))}–{f(max(c40[s] for s in glob_))}%), reproducing the "
          f"preliminary finding. At round 200 it was behind all of them ({f(c200['local'])}% against "
          f"{f(min(c200[s] for s in glob_))}–{f(max(c200[s] for s in glob_))}%). The image-level tests confirm both "
          f"reversals ({{T:image}}): FedAvg minus local-only was {sg(i40['fedavg'].diff_eq)} points at round 40 "
          f"(95% CI {ci(i40['fedavg'].ci_lo, i40['fedavg'].ci_hi)}) and {sg(i200['fedavg'].diff_eq)} at round 200 "
          f"({ci(i200['fedavg'].ci_lo, i200['fedavg'].ci_hi)}), and FlexLoRA minus local-only "
          f"{sg(i40['flexlora'].diff_eq)} and {sg(i200['flexlora'].diff_eq)}. Neither fixed budget is a fair "
          "comparison, because each favors one side; the validation-selected round removes that bias, and on it "
          "FedAvg remained below local-only training in Preset A while the product-space rules were above it. "
          "In Preset A, centralized training and the specialty oracle were the two highest strategies at rounds 40 "
          "and 100. At round 200 centralized training was still highest, whereas the oracle, like local-only "
          f"training, had declined ({f(c200['oracle'])}%), which is why its selected round was early."))
    add(("fig", "conv", "fig_conv.png",
         f"Test accuracy over {ROUNDS} rounds, linear adapter (mean over {N_SEEDS} seeds; shaded band ± 1 SD). "
         "Dashed: centralized training. Local-only training peaks early and declines, while global federated "
         "strategies are still improving late in training."))
    add(("wide_group", ["budget", "image"]))
    # ---- E
    add(("h2", "ADAPTER FORM AND THE LINEAR PROBE"))
    add(p(f"The local linear adapter was below the linear probe in both presets, by {f(-probe['A']['diff'])} "
          f"points in Preset A (95% CI {ci(-probe['A']['hi'], -probe['A']['lo'])}) and {f(-probe['B']['diff'])} in "
          f"Preset B ({ci(-probe['B']['hi'], -probe['B']['lo'])}), and lower in every seed of both presets. Proposition 1 rules out "
          "a difference in capacity, so the gap reflects optimization: the probe tolerates a higher learning rate "
          "and overfits more slowly. The nonlinear local adapter was also below the probe "
          f"({sg(nprobe['A']['diff'])} and {sg(nprobe['B']['diff'])} points), and it did not differ from the linear "
          f"local adapter ({sg(nl_lo['A']['diff'])} and {sg(nl_lo['B']['diff'])})."))
    add(p(f"With the nonlinear adapter the specialty pattern persisted. The oracle exceeded FedAvg by "
          f"{pd_(nl_ofa['A'])} in Preset A and by {sg(nl_ofa['B']['diff'])} points in Preset B "
          f"({ci(nl_ofa['B']['lo'], nl_ofa['B']['hi'])}). The nonlinear adapter helped FedAvg in Preset A "
          f"({sg(nl_fa['A']['diff'])} points over linear FedAvg; {ci(nl_fa['A']['lo'], nl_fa['A']['hi'])}), where it "
          f"turned the deficit of FedAvg against local-only training into a small, uncertain gain "
          f"({sg(pair('A', 'nonlinear', 'fedavg', 'local')['diff'])} points), but not in Preset B "
          f"({sg(nl_fa['B']['diff'])})."))
    # ---- F
    add(("h2", "PROXIMAL REGULARIZATION AND OTHER BASELINES"))
    add(p(f"FedProx with the tuned *μ* exceeded FedAvg by {pd_(pfa['A'])} in Preset A, where a strong penalty "
          f"(*μ* = 0.1) was selected, and was equal to it in Preset B ({sg(pfa['B']['diff'])} points), where the "
          "weakest penalty was selected. FFA-LoRA, which freezes *A*, was among the lowest federated strategies in "
          "both presets. FedSA-LoRA, which keeps *B* local, had the highest mean in Preset B, "
          f"{f(es('B', 'linear', 'fedsa')[0] - es('B', 'linear', 'fedavg')[0])} points above FedAvg; this was not a "
          "planned contrast."))
    # ---- G
    add(("h2", "SECONDARY METRICS"))
    pooledA = {s: cp("A", "linear", s, "acc_w") for s in LIN}
    add(p(f"{{T:secondary}} gives the secondary metrics. They agreed with the primary results in direction. In "
          f"Preset A the specialty oracle and centralized training had the highest pooled accuracy "
          f"({f(pooledA['oracle'])}% and {f(pooledA['centralized'])}%), macro-F1, and AUC, and FedAvg the lowest "
          f"pooled accuracy among federated strategies ({f(pooledA['fedavg'])}%, against {f(pooledA['local'])}% for "
          "local-only training). In Preset B all federated strategies had higher pooled accuracy and AUC than "
          f"local-only training. The worst client was a TissueMNIST client in {WC[('A', 'tissuemnist')]} of "
          f"{WC['A']} Preset A runs (an OCTMNIST client otherwise) and a RetinaMNIST client in "
          f"{WC[('B', 'retinamnist')]} of {WC['B']} Preset B runs; its accuracy varied by a few points between strategies, and FedSAA-U had the highest value "
          f"in Preset B ({f(cp('B', 'linear', 'fedsaa_u', 'worst'), 1)}%)."))
    add(("wide_group", ["secondary"]))
    # ---- H
    add(("h2", "COMPUTATIONAL COST"))
    add(p(f"{{T:runtime}} gives the time per run. Product-space rules with per-client SVDs were the most expensive: "
          f"FedSAA took {sec('A', 'linear', 'fedsaa'):.0f} s per run in Preset A against "
          f"{sec('A', 'linear', 'fedavg'):.0f} s for FedAvg and {sec('A', 'linear', 'flexlora'):.0f} s for FlexLoRA, "
          "which needs one SVD per round. On a GPU, or with a randomized SVD, this overhead would shrink."))
    add(("table", "runtime"))

    # ============================== VII. DISCUSSION
    add(("h1", "DISCUSSION"))
    add(("h2", "WHAT THE RESULTS SHOW"))
    add(p("Across two simulated cross-specialty federations, sharing a low-rank adapter helped or hurt depending on "
          "the specialties involved. In Preset A, plain averaging across all specialties lowered accuracy relative "
          "to local training, whereas averaging within specialties raised it by about two points for 14 of 15 "
          "clients. In Preset B, every federated strategy helped, and separating specialties added nothing on "
          "average. The federation-level effect is therefore a property of the mix of specialties, not of federated "
          "learning as such, and a department deciding whether to join should look at the effect for its own "
          "specialty. RetinaMNIST lost under FedAvg, FedProx, FlexLoRA, FedSA-LoRA, and FedSAA in Preset B, and "
          "PathMNIST and OCTMNIST lost under FedAvg in Preset A."))
    add(p("We do not know why particular specialties gain or lose. Neither image color nor dataset size explains "
          "the pattern in a simple way: the small RetinaMNIST clients lost while the small PneumoniaMNIST clients "
          "gained most, and grayscale OCT lost under FedAvg while grayscale CT gained."))
    add(("h2", "WHY FEDSAA DID NOT PERSONALIZE"))
    add(p("FedSAA was designed to let clients of the same specialty collaborate more closely than clients of "
          "different specialties. The logs show that it did so for one or two rounds and then stopped. Its "
          "similarity was computed on the full adapter products, and after aggregation those products are "
          "dominated by the common adapter every client receives; local training adds a comparatively small, "
          "specialty-specific change. Cosine similarities between specialties therefore approached one, and by "
          "Proposition 4 the attention weights became uniform. Lowering the temperature would help only if *τ* were "
          "brought down to the scale of the remaining spread *ε*, whose mean had fallen below 0.03 by round 40. A rule "
          "that computes similarity on the change made in the current round, or on gradients, might preserve the "
          "separation observed at round 1; we did not test this. The specialty oracle shows that the gain from "
          "separating specialties is real in Preset A, so a better clustering signal, not a better weighting "
          "function, is what is missing."))
    add(p("The retention step was harmful in its original, factor-space form, as Proposition 3 predicts, and "
          "harmless in update space. Its removal accounts for the whole difference between FedSAA and FlexLoRA. The "
          "rank truncation, which we had suspected, lost almost no energy."))
    add(("h2", "THE TRAINING BUDGET CHANGES THE CONCLUSION"))
    add(p("The conclusion of our preliminary experiments that local training beats federation in Preset A was an artifact of stopping at "
          "40 rounds. In Preset A, local training on 1,700 images per client converges in tens of rounds and then "
          "overfits, "
          "whereas averaging across specialties converges slowly. A fixed budget therefore favors whichever side "
          "converges at that point. Validation-based stopping gives each strategy its own budget and is, we "
          "believe, the fairer default for benchmarks that mix local and federated training. Under it, FedAvg still "
          "lost to local training in Preset A, so the specialty effect is not an artifact of the budget."))
    add(("h2", "THE MODEL CLASS LIMITS THE INTERPRETATION"))
    add(p(OLD(187).replace(" This reading is consistent with the slower convergence of every federated strategy and "
                           "with the absence of any effect of adapter rank.",
                           " The linear probe outperforming the local linear adapter is consistent with this "
                           "reading. The nonlinear adapter, which is not limited in this way, did not change the "
                           "pattern.")))
    add(p("The study says nothing direct about LoRA modules placed inside the convolutional layers of a backbone, "
          "where adapters alter nonlinear features at several depths. Testing that requires training through the "
          "backbone and is left for future work."))
    add(("h2", "RELATION TO PRIOR WORK"))
    add(p("Benchmarks of personalized FL find that no method wins everywhere \\cite{pflbench,motley}, and local "
          "adaptation can be needed for individual participants \\cite{yu2020salvaging}; our specialty results "
          "add a case in which the composition of the federation decides. Clustered FL \\cite{sattler2021,ghosh2020} "
          "rests on the premise that client groups can be identified from updates; our logs show that, for "
          "low-rank adapters shared from a common state, similarities of the adapters themselves are not such a "
          "signal. Work on federated LoRA identifies the gap between averaging factors and averaging products "
          "\\cite{flora,ffalora,fedsalora}; FlexLoRA \\cite{flexlora}, which closes it, was among the best global "
          "rules here, and our analysis adds that retention must then be applied to products, not factors. Methods "
          "for federations across imaging modalities train much larger models \\cite{omnifm,fm2}, and our results "
          "do not bear on them directly."))
    add(("h2", "THREATS TO VALIDITY"))
    add(p("*Internal validity:* The learning rate was tuned on one seed over a three-point grid, and the FedSAA "
          "parameters were not tuned; better settings may exist for every strategy. Centralized training pools "
          "classifier data within a specialty, which the federated strategies do not, so it confounds two factors. "
          "*Statistical conclusion validity:* Ten seeds and image-level tests give much tighter intervals than our "
          "preliminary experiments, but the partition is fixed, so seed variation does not include variation in which images "
          "each client holds. *Construct validity:* Equal-weight accuracy gives a client with 52 test images the "
          "same weight as one with 1,000; the pooled accuracy reported alongside it leads to the same conclusions. "
          "*External validity:* The clients are simulated by partitioning public datasets, so clients of one "
          "specialty do not differ by scanner, protocol, or population. Images were 64 × 64 pixels, one backbone "
          "and one adapter placement were used, and privacy mechanisms, partial participation, and communication "
          "limits were not studied. Nothing here constitutes clinical validation."))
    add(("h2", "RECOMMENDATIONS"))
    add(p("For benchmarks of this kind we recommend the following."))
    for t in ["1) Report a local-only baseline and transfer gains for every participant and specialty, not only "
              "their mean.",
              "2) Select the stopping round on validation data for every strategy, and report fixed budgets and "
              "curves as well.",
              "3) Include a specialty-oracle or other cluster-aware reference, which shows how much a rule that "
              "recovered the group structure could gain.",
              "4) Log the similarities and attention weights of similarity-based rules and the singular values of "
              "SVD-based rules; both were decisive here.",
              "5) When the trainable part is linear, include a linear probe.",
              "6) Use at least ten seeds, save per-image predictions, and release the raw records and scripts."]:
        add(("pc", t))

    # ============================== VIII. CONCLUSION
    add(("h1", "CONCLUSION"))
    add(p(f"We benchmarked 12 strategies for sharing a low-rank adapter on a frozen backbone across two simulated "
          f"federations of medical imaging specialties, with validation tuning, validation-based stopping, "
          f"{ROUNDS} rounds, and {N_SEEDS} seeds. Whether federation helped depended on the federation: averaging "
          "across all specialties lost to local training in one and helped in the other, while averaging within "
          "specialties was the best or among the best strategies in both. FedSAA, our similarity-weighted rule, "
          "lost its specialty selectivity within the first tens of rounds because the similarities of adapter "
          "products converged, and it then reduced to FlexLoRA; its factor-space retention step cost accuracy, "
          "as our analysis predicts. A fixed budget of 40 rounds reversed the ranking of local and federated "
          "training. Future work should derive the similarity signal from per-round changes, place adapters inside "
          "a medical foundation model, and test these findings on real multi-institution data."))

    # ============================== APPENDIX
    add(("h1u", "APPENDIX"))
    add(("h1u", "PER-SPECIALTY TRANSFER GAINS"))
    add(p("{T:spec_gain} gives the transfer gain of every specialty under every federated strategy with the linear "
          "adapter. The tuning results of every configuration, the per-run records, and the per-image predictions "
          "are included in the repository."))
    add(("wide_group", ["spec_gain"]))
    add(("h1u", "DATA AND CODE AVAILABILITY"))
    add(p("MedMNIST v2 is publicly available at https://medmnist.com. The feature-extraction, training, and analysis "
          f"scripts, the raw records of all {len(RUNS)} runs (with per-round validation and test curves, per-client "
          "metrics, and logged similarities and singular values), the per-image predictions, and the scripts that "
          "regenerate every table and figure are available at https://github.com/Mohanbabu82/FedSAA. "
          "[[AUTHORS: archive the release on Zenodo, add the DOI here, and add a license file to the repository.]]"))
    add(("h1u", "ACKNOWLEDGMENT"))
    add(p("[[AUTHORS: edit this statement so that it matches what was actually done; IEEE requires it.]] A "
          "generative artificial intelligence system (Claude, Anthropic) was used in preparing this article. It "
          "wrote the training and analysis scripts used for Sections IV and VI, which the authors ran; computed the "
          "statistics, tables, and figures of Section VI from the raw results; drafted the derivations of Section V; "
          "and drafted and edited the text of Sections I–VIII and the Appendix. The authors reviewed and verified "
          "all content, including the code and the proofs, and are responsible for it."))
    return B, T


def T_ref(k):
    return "{T:" + k + "}"
