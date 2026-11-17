# -*- coding: utf-8 -*-
"""Long (about 17-page) IEEE Access version: extends content.py with a problem formulation,
a mathematical analysis of the aggregation rule, and the full per-specialty, per-client and
pairwise results. Tables with numbers are built from the analysis CSV files, not typed.

Extra markup resolved by the build script:
    \\cite{key1,key2}   numbered by first appearance
    {F:key} {T:key}     figure / table cross-references, numbered in order of placement
"""
import os
import re

import pandas as pd

import content as C
from refs import OLD_KEYS

TITLE, AUTHORS, AFFILIATION, CORRESPONDING, SUPPORT = C.TITLE, C.AUTHORS, C.AFFILIATION, C.CORRESPONDING, C.SUPPORT
ABSTRACT, INDEX_TERMS, BIOS = C.ABSTRACT, C.INDEX_TERMS, C.BIOS

NICE = {"pathmnist": "PathMNIST", "dermamnist": "DermaMNIST", "bloodmnist": "BloodMNIST",
        "octmnist": "OCTMNIST", "tissuemnist": "TissueMNIST", "organamnist": "OrganAMNIST",
        "pneumoniamnist": "PneumoniaMNIST", "retinamnist": "RetinaMNIST",
        "breastmnist": "BreastMNIST", "organcmnist": "OrganCMNIST"}
LABEL = {"centralized": "Centralized", "local_only": "Local-only", "fedavg": "FedAvg",
         "fedprox": "FedProx", "fedsaa": "FedSAA"}
METHODS = ["centralized", "local_only", "fedavg", "fedprox", "fedsaa"]


# ------------------------------------------------------------------ reuse of the short text
def _keyify(t):
    """Turn the numeric citations of the short version into \\cite{key} commands."""
    def rng(m):
        a, b = int(m.group(1)), int(m.group(2))
        return "\\cite{" + ",".join(OLD_KEYS[k - 1] for k in range(a, b + 1)) + "}"
    t = re.sub(r"\[(\d+)\]–\[(\d+)\]", rng, t)
    t = re.sub(r"\[(\d+)\], \[(\d+)\]",
               lambda m: "\\cite{%s,%s}" % (OLD_KEYS[int(m.group(1)) - 1], OLD_KEYS[int(m.group(2)) - 1]), t)
    t = re.sub(r"\[(\d+)\]", lambda m: "\\cite{%s}" % OLD_KEYS[int(m.group(1)) - 1], t)
    return t


_XREF = [
    ("(Table I)", "({T:composition})"),
    ("Table II lists the configuration.", "{T:config} lists the configuration."),
    ("as Table III shows", "as {T:main} shows"),
    ("(Table IV)", "({T:contrasts})"),
    ("(Table V)", "({T:sensitivity})"),
    ("The ranking in Table III", "The ranking in {T:main}"),
    ("(Fig. 1)", "({F:specialty})"),
    ("(Fig. 2)", "({F:conv})"),
    ("Section III-D", "Section IV-D"),
    ("figures of Section IV", "figures of Section VI"),
]
_seen = set()


def _old_text(t):
    t = _keyify(t)
    for a, b in _XREF:
        if a in t:
            t = t.replace(a, b)
            _seen.add(a)
    return t


_OLD_P = [_old_text(it[1]) for it in C.BODY if it[0] in ("p", "pc")]
assert len(_seen) == len(_XREF), set(a for a, _ in _XREF) - _seen


def O(prefix):
    """The paragraph of the short version that starts with prefix."""
    hits = [t for t in _OLD_P if t.startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


def p(t):
    return ("p", t)


def pc(t):
    return ("pc", t)


# ------------------------------------------------------------------ number formatting
def f(v, nd=2):
    return f"{v:.{nd}f}".replace("-", "−")


def sg(v, nd=2):
    return ("+" if v > 0 else "") + f(v, nd)


def pv(v):
    return "<0.001" if v < 0.0005 else f"{v:.3f}"


# ------------------------------------------------------------------ tables from the analysis files
def data_tables(tdir):
    T = {}
    rd = lambda name: pd.read_csv(os.path.join(tdir, name))
    t1, t2, t3, t4 = rd("table1_main_results.csv"), rd("table2_pairwise_tests.csv"), \
        rd("table3_per_specialty.csv"), rd("table4_delta_vs_local.csv")
    t6, t7, t7b = rd("table6_convergence_summary.csv"), rd("table7_client_level.csv"), rd("table7b_client_vs_local.csv")

    # per-specialty accuracy
    rows = []
    for pr in ("A", "B"):
        rows.append(("group", f"Preset {pr}"))
        sub = t3[t3.preset == pr]
        for s in dict.fromkeys(sub.specialty):
            r = [NICE[s]]
            for m in METHODS:
                x = sub[(sub.specialty == s) & (sub.method == m)].iloc[0]
                r.append(f"{f(x.acc_mean)} ± {f(x.acc_sd)}")
            rows.append(r)
    T["spec_acc"] = {"title": "Accuracy of Each Specialty at Round 40 (%)", "widths": [1800] + [1600] * 5,
                     "size": 16, "header": ["Specialty"] + [LABEL[m] for m in METHODS],
                     "align": ["left"] + ["center"] * 5, "rows": rows,
                     "note": "Values are the mean ± sample SD over three seeds of the mean accuracy of the three "
                             "clients of a specialty."}

    # per-specialty transfer gain with CI
    rows, excl = [], 0
    for pr in ("A", "B"):
        rows.append(("group", f"Preset {pr}"))
        sub = t4[t4.preset == pr]
        for s in dict.fromkeys(sub.specialty):
            r = [NICE[s]]
            for m in ("centralized", "fedavg", "fedprox", "fedsaa"):
                x = sub[(sub.specialty == s) & (sub.method == m)].iloc[0]
                r.append(f"{sg(x.delta_vs_local)} ({f(x.ci_lo)}, {f(x.ci_hi)})")
                excl += int(x.ci_lo > 0 or x.ci_hi < 0)
            rows.append(r)
    T["spec_delta"] = {"title": "Transfer Gain of Each Specialty Relative to Local-Only Training",
                       "widths": [1800] + [2000] * 4, "size": 16,
                       "header": ["Specialty", "Centralized", "FedAvg", "FedProx", "FedSAA"],
                       "align": ["left"] + ["center"] * 4, "rows": rows,
                       "note": "Values are the mean difference in accuracy from local-only training in percentage "
                               "points, with the 95% confidence interval over three seeds in parentheses."}
    T["_n_excl"] = excl

    # convergence checkpoints
    rows = []
    for pr in ("A", "B"):
        rows.append(("group", f"Preset {pr}"))
        for m in METHODS:
            x = t6[(t6.preset == pr) & (t6.method == m)].iloc[0]
            rows.append([LABEL[m], f(x.mean_at_round_10, 1), f(x.mean_at_round_20, 1), f(x.mean_at_round_30, 1),
                         f(x.final_mean, 1), sg(x.slope_last10_pts_per_round)])
    T["conv"] = {"title": "Accuracy at Four Checkpoints and Late Slope", "widths": [1150, 700, 700, 700, 700, 750],
                 "size": 15, "header": ["Strategy", "Round 10", "Round 20", "Round 30", "Round 40", "Slope"],
                 "align": ["left"] + ["center"] * 5, "rows": rows,
                 "note": "Accuracy in %, mean over three seeds. Slope is the least-squares slope over rounds 31–40 "
                         "in percentage points per round."}

    # client-level summary
    rows = []
    for pr in ("A", "B"):
        rows.append(("group", f"Preset {pr}"))
        for m in ("centralized", "fedavg", "fedprox", "fedsaa"):
            x = t7b[(t7b.preset == pr) & (t7b.method == m)].iloc[0]
            rows.append([LABEL[m], str(int(x.clients_above_local)), str(int(x.clients_below_local)),
                         sg(x.median_delta, 1), sg(x.min_delta, 1), sg(x.max_delta, 1)])
    T["clients"] = {"title": "Client-Level Transfer Gains", "widths": [1200, 650, 650, 750, 750, 700], "size": 15,
                    "header": ["Strategy", "Above", "Below", "Median", "Min.", "Max."],
                    "align": ["left"] + ["center"] * 5, "rows": rows,
                    "note": "Above and Below count the clients, out of 15, whose seed-averaged accuracy is above or "
                            "below local-only training. Median, Min., and Max. are transfer gains in percentage "
                            "points."}

    # run time
    rows = []
    for m in METHODS:
        a = t1[(t1.preset == "A") & (t1.method == m)].iloc[0].seconds_mean
        b = t1[(t1.preset == "B") & (t1.method == m)].iloc[0].seconds_mean
        rows.append([LABEL[m], f(a, 0), f(b, 0)])
    T["runtime"] = {"title": "Wall-Clock Time of One Run", "widths": [1700, 1500, 1500], "size": 15,
                    "header": ["Strategy", "Preset A (s)", "Preset B (s)"], "align": ["left", "center", "center"],
                    "rows": rows, "note": "Mean over three seeds; 40 rounds on one GPU, including evaluation after "
                                          "every round."}

    # full pairwise comparisons, one table per metric
    spec = [("acc", "Accuracy", 2, "pairs_acc"), ("f1", "Macro-F1", 3, "pairs_f1"),
            ("auc", "AUC", 4, "pairs_auc"), ("worst_client", "Worst-Client Accuracy", 2, "pairs_worst")]
    for met, name, nd, key in spec:
        a = t2[(t2.metric == met) & (t2.preset == "A")].reset_index(drop=True)
        b = t2[(t2.metric == met) & (t2.preset == "B")].reset_index(drop=True)
        rows = []
        for k in range(len(a)):
            x, y = a.iloc[k], b.iloc[k]
            assert (x.a, x.b) == (y.a, y.b)
            rows.append([f"{LABEL[x.a]} − {LABEL[x.b]}",
                         sg(x.mean_diff, nd), f"{f(x.ci_lo, nd)} to {f(x.ci_hi, nd)}", pv(x.p_t), pv(x.p_holm),
                         sg(y.mean_diff, nd), f"{f(y.ci_lo, nd)} to {f(y.ci_hi, nd)}", pv(y.p_t), pv(y.p_holm)])
        T[key] = {"title": f"All Pairwise Comparisons: {name}",
                  "widths": [2200, 800, 1600, 650, 750, 800, 1600, 650, 750], "size": 15,
                  "header": ["Contrast", "Diff. (A)", "95% CI (A)", "*p* (A)", "Adj. *p* (A)",
                             "Diff. (B)", "95% CI (B)", "*p* (B)", "Adj. *p* (B)"],
                  "align": ["left"] + ["center"] * 8, "rows": rows,
                  "note": "Diff. is the first strategy minus the second"
                          + (", in percentage points" if met in ("acc", "worst_client") else "")
                          + ". (A) and (B) are the two presets. Two-sided paired *t*-test over three seeds; Adj. *p* "
                            "is Holm-adjusted across the ten contrasts of a preset."}

    # every client
    rows = []
    for pr in ("A", "B"):
        rows.append(("group", f"Preset {pr}"))
        for x in t7[t7.preset == pr].itertuples():
            rows.append([str(x.client + 1), NICE[x.specialty], f"{int(x.test_n):,}"]
                        + [f(getattr(x, m)) for m in METHODS])
    T["client_full"] = {"title": "Accuracy of Every Client at Round 40 (%)",
                        "widths": [800, 1900, 1100] + [1200] * 5, "size": 15,
                        "header": ["Client", "Specialty", "Test images"] + [LABEL[m] for m in METHODS],
                        "align": ["center", "left", "center"] + ["center"] * 5, "rows": rows,
                        "note": "Mean over three seeds."}
    return T


def static_tables():
    import math
    T = {k: dict(v) for k, v in C.TABLES.items()}
    rows = []
    for tau in (0.1, 0.3, 0.5, 1.0, 2.0):
        for s in (0.0, 0.5, 0.9):
            z = math.exp(1 / tau) + 2 * math.exp(s / tau) + 12
            other = 12 / z
            rows.append([f"{tau:g}", f"{s:g}", f(math.exp(1 / tau) / z), f(2 * math.exp(s / tau) / z), f(other),
                         f(0.5 * other + 0.5 * 12 / 15)])
    T["selfweight"] = {
        "title": "Attention Weights Under an Idealized Similarity Structure", "widths": [600, 600, 800, 800, 800, 1100],
        "size": 15, "header": ["*τ*", "*s*", "Self", "Same", "Other", "Other, blended"],
        "align": ["center"] * 6, "rows": rows,
        "note": "Fifteen clients; the two same-specialty peers have similarity *s* with the client and the 12 "
                "other-specialty clients have similarity 0. Self, Same, and Other are shares of the personalized "
                "aggregate. Other, blended is the share of other specialties after blending with the global mean "
                "(*λ* = 0.5)."}
    T["notation"] = {
        "title": "Notation", "widths": [1150, 3550], "size": 15, "header": ["Symbol", "Meaning"],
        "align": ["left", "left"],
        "rows": [
            ["*N*", "Number of clients (15)"],
            ["*D*~*i*~, *n*~*i*~", "Dataset and number of training images of client *i*"],
            ["*x*, *d*", "Frozen feature vector and its dimension (256)"],
            ["*A*, *B*", "Adapter factors, of size *r* × *d* and *d* × *r*"],
            ["*r*, *α*", "Adapter rank (16) and scaling (32)"],
            ["*U*~*i*~", "Adapter update of client *i*, equal to *B*~*i*~*A*~*i*~"],
            ["*S*~*ij*~", "Cosine similarity between *U*~*i*~ and *U*~*j*~"],
            ["*w*~*ij*~", "Attention weight of client *j* in the aggregate of client *i*"],
            ["*τ*", "Attention temperature (0.5)"],
            ["*λ*", "Weight of the global mean in the blend (0.5)"],
            ["*ρ*", "Retention coefficient (0.3)"],
            ["*μ*", "FedProx proximal weight (0.01)"],
            ["*E*, *T*", "Local epochs per round (3) and number of rounds (40)"],
            ["Δ~*i*~(*m*)", "Transfer gain of strategy *m* for client *i*"],
        ], "note": None}
    T["related"] = {
        "title": "Federated Low-Rank Adaptation Methods Compared With FedSAA",
        "widths": [1500, 1900, 3100, 1900, 1400], "size": 15,
        "header": ["Method", "Communicated", "Server aggregation", "Kept local", "Main setting"],
        "align": ["left"] * 5,
        "rows": [
            ["FedIT \\cite{fedit}", "*A* and *B*", "Weighted average of *A* and of *B* separately", "Nothing", "Language models"],
            ["FFA-LoRA \\cite{ffalora}", "*B* only; *A* is frozen", "Average of *B*", "Nothing", "Language models, privacy"],
            ["FedSA-LoRA \\cite{fedsalora}", "*A* only", "Average of *A*", "*B*", "Language models"],
            ["FLoRA \\cite{flora}", "*A* and *B*, any rank", "Stacking of the client modules", "Nothing", "Language models"],
            ["HetLoRA \\cite{hetlora}", "*A* and *B*, client-specific rank", "Sparsity-weighted average after zero-padding", "Choice of rank", "On-device language models"],
            ["FlexLoRA \\cite{flexlora}", "*A* and *B*, client-specific rank", "Weighted average of full-size products, then SVD", "Choice of rank", "Language models"],
            ["FedOPAL \\cite{fedopal}", "Global adapter", "Average of the global adapter", "Orthogonal personal adapter", "Medical vision models"],
            ["FedSAA (this work)", "*A* and *B*", "Per-client similarity-weighted average of products, blended with the global mean, then SVD", "Classifier; part of the adapter through retention", "Medical imaging, frozen backbone"],
        ], "note": None}
    return T


ALGORITHM = ("Algorithm 1: FedSAA", [
    (0, "**Input:** clients 1, …, *N*; rounds *T*; local epochs *E*; temperature *τ*; blend weight *λ*; retention *ρ*; rank *r*"),
    (0, "1: initialize the factors *A*~*i*~, *B*~*i*~ and the classifier of every client *i*"),
    (0, "2: **for** *t* = 1, …, *T* **do**"),
    (1, "3: **for** every client *i* in parallel **do**"),
    (2, "4: train *A*~*i*~, *B*~*i*~ and the classifier for *E* epochs on *D*~*i*~; send *A*~*i*~, *B*~*i*~ to the server"),
    (1, "5: **end for**"),
    (1, "6: *U*~*i*~ ← *B*~*i*~*A*~*i*~ for all *i*; *S*~*ij*~ ← cos(*U*~*i*~, *U*~*j*~) for all *i*, *j*"),
    (1, "7: *G* ← mean of *U*~1~, …, *U*~*N*~"),
    (1, "8: **for** every client *i* **do**"),
    (2, "9: *w*~*ij*~ ← softmax over *j* of *S*~*ij*~/*τ*"),
    (2, "10: *Ũ*~*i*~ ← (1 − *λ*) Σ~*j*~ *w*~*ij*~*U*~*j*~ + *λG*"),
    (2, "11: (*A*′~*i*~, *B*′~*i*~) ← factors of the rank-*r* truncated SVD of *Ũ*~*i*~"),
    (2, "12: *A*~*i*~ ← (1 − *ρ*)*A*′~*i*~ + *ρA*~*i*~; *B*~*i*~ ← (1 − *ρ*)*B*′~*i*~ + *ρB*~*i*~; return both to client *i*"),
    (1, "13: **end for**"),
    (0, "14: **end for**"),
])


# ------------------------------------------------------------------ the document
def build_body(tdir):
    T = static_tables()
    T.update(data_tables(tdir))
    n_excl = T.pop("_n_excl")
    E = C.EQUATIONS

    B = []
    add = B.append
    ext = B.extend

    # ============================== I. INTRODUCTION
    add(("h1", "INTRODUCTION"))
    ext([p(O("Medical images are difficult")), p(O("The main obstacle to FL")), p(O("Parameter-efficient fine-tuning makes")),
         p(O("Here we test this directly"))])
    add(p("The study is organized around four research questions. RQ1: Does federating a shared low-rank adapter "
          "improve client accuracy over local-only training when the clients come from different specialties? RQ2: "
          "Does similarity-weighted personalized aggregation improve on FedAvg and FedProx? RQ3: Which specialties "
          "and clients gain or lose under federation? RQ4: How do the answers depend on the training budget and on "
          "the hyperparameters of FedSAA?"))
    add(p(O("The outcome is negative for the proposed method")))
    add(p("The rest of this article is organized as follows. Section II reviews related work. Section III states the "
          "problem and the notation. Section IV describes the data, model, training strategies, and statistical "
          "analysis. Section V analyzes the aggregation rule mathematically. Section VI reports the results, "
          "Section VII discusses them and their limitations, and Section VIII concludes. The Appendix gives the "
          "complete pairwise comparisons and the client-level results."))

    # ============================== II. RELATED WORK
    add(("h1", "RELATED WORK"))
    add(("h2", "FEDERATED BENCHMARKS IN MEDICAL IMAGING"))
    add(p(O("FLamby provides cross-silo healthcare datasets")))
    add(p("Simulated federations are most often built by partitioning one labeled dataset with a Dirichlet prior "
          "over class proportions \\cite{hsu2019}, which produces label skew while every client still solves the same "
          "task. Experimental studies of such partitions report that no single algorithm dominates across types of "
          "heterogeneity \\cite{li2022silos}. The cross-specialty setting studied here is different in kind. Clients "
          "differ in modality, label space, and task, so there is no common output layer to average, and the only "
          "candidate for sharing is the feature transformation."))
    add(("h2", "HETEROGENEITY AND PERSONALIZATION"))
    add(p(O("FedProx adds a proximal term")))
    add(p("Other approaches treat personalization as multi-task learning \\cite{smith2017}, as meta-learning of an "
          "initialization that adapts quickly to each client \\cite{fallah2020}, or as bi-level optimization with a "
          "Moreau-envelope penalty \\cite{dinh2020}. FedRoD decouples a generic predictor from a personalized head "
          "\\cite{chen2022fedrod}. FedNova shows that unequal amounts of local work bias the averaged objective and "
          "corrects for it by normalizing the updates \\cite{wang2020fednova}. A general overview of these "
          "challenges is given in \\cite{li2020spm}."))
    add(("h2", "FEDERATED LOW-RANK ADAPTATION"))
    add(p("Parameter-efficient fine-tuning adapts a pretrained network by training a small number of added "
          "parameters. Adapter modules insert small bottleneck layers into a frozen network \\cite{houlsby2019}, and "
          "LoRA reparameterizes the update of a weight matrix as a product of two low-rank factors "
          "\\cite{hu2022lora}. In FL these methods reduce communication by orders of magnitude, and they have been "
          "compared systematically for language models \\cite{zhang2023fedpetuning}. Starting from pretrained "
          "weights also makes federated optimization less sensitive to heterogeneity \\cite{nguyen2023}."))
    add(p(O("FedIT averages the LoRA factors") + " {T:related} summarizes how these methods differ in what is "
          "communicated, where aggregation takes place, and what stays local."))
    add(("wide_group", ["related"]))
    add(("h2", "NEGATIVE TRANSFER AND LINEAR PROBING"))
    add(p("Negative transfer occurs when knowledge from a source task lowers performance on a target task relative "
          "to learning the target alone \\cite{pan2010,wang2019neg}. In multi-task learning it is linked to "
          "conflicting gradients between tasks \\cite{yu2020pcgrad}, and whether tasks help each other depends on "
          "which tasks are grouped \\cite{standley2020}. In medical imaging, the benefit of ImageNet pretraining "
          "has been found to be smaller than commonly assumed and to depend on model size and data regime "
          "\\cite{raghu2019}. Our model trains a linear map on frozen features, which places it close to linear "
          "probing \\cite{alain2016,kornblith2019}. Linear probing keeps pretrained features intact, whereas "
          "fine-tuning can distort them \\cite{kumar2022}. The adapter studied here changes the features only "
          "through a linear transformation, a point that Section V makes precise."))
    add(("h2", "REPORTING COMPARISONS"))
    add(p(O("Differences between learning algorithms") + " Classical guidance recommends paired tests on matched "
          "runs and warns against reading significance from a handful of comparisons "
          "\\cite{dietterich1998,demsar2006}, and the American Statistical Association cautions against basing "
          "conclusions on whether a *p* value passes a threshold \\cite{wasserstein2016}."))

    # ============================== III. PRELIMINARIES
    add(("h1", "PRELIMINARIES AND PROBLEM FORMULATION"))
    add(("h2", "FEDERATED OPTIMIZATION"))
    add(p("Consider *N* clients. Client *i* holds a dataset *D*~*i*~ of *n*~*i*~ labeled images and has the empirical "
          "risk"))
    add(("eq", r"F_i(\theta) = \frac{1}{n_i}\sum_{(x,y)\in D_i}\ell\big(f(x;\theta),\,y\big)"))
    add(pc("where *f* is the model with parameters *θ* and *ℓ* is the cross-entropy loss. Standard FL seeks a single "
           "parameter vector that minimizes the weighted average risk \\cite{mcmahan2017},"))
    add(("eq", r"\min_{\theta}\;F(\theta) = \sum_{i=1}^{N}\frac{n_i}{n}\,F_i(\theta)"))
    add(pc("where *n* is the total number of training images. In each round of FedAvg, every client starts from the "
           "current global parameters, runs *E* local epochs, and returns its updated parameters. The server then "
           "replaces the global parameters with their weighted average,"))
    add(("eq", r"\theta^{t+1} = \sum_{i=1}^{N}\frac{n_i}{n}\,\theta_i^{t+1}"))
    add(pc("where the superscript counts rounds. FedProx changes only the local objective \\cite{li2020fedprox}, to"))
    add(("eq", r"F_i(\theta) + \frac{\mu}{2}\,\lVert\theta - \theta^{t}\rVert^{2}"))
    add(pc("which penalizes movement away from the global parameters. When the client distributions differ, the "
           "minimizers of the local risks differ, the local iterates drift toward them during the *E* epochs, and "
           "the average need not be a good point for any client "
           "\\cite{zhao2018,karimireddy2020,wang2020fednova}."))
    add(("h2", "SHARED AND PRIVATE PARAMETERS"))
    add(p("In the cross-specialty setting clients cannot share a full model, because their label spaces differ. The "
          "parameters are therefore split into a shared part *ψ*, here the adapter, and a private part *φ*~*i*~, here "
          "the classifier of client *i* \\cite{arivazhagan2019,collins2021}. The federated objective becomes"))
    add(("eq", r"\min_{\psi,\,\phi_1,\dots,\phi_N}\;\sum_{i=1}^{N}\frac{n_i}{n}\,F_i(\psi,\phi_i)"))
    add(pc("and only *ψ* is communicated. Local-only training removes the coupling by giving every client its own "
           "copy of *ψ*. Centralized training optimizes the same objective with all data in one place. Personalized "
           "aggregation, as in FedSAA, lies in between: the server returns a different adapter to each client, "
           "computed from the adapters of all clients."))
    add(("h2", "LOW-RANK ADAPTATION"))
    add(p("LoRA replaces the update of a frozen weight matrix *W*~0~ by a low-rank product \\cite{hu2022lora},"))
    add(("eq", r"W = W_0 + \frac{\alpha}{r}\,BA"))
    add(pc("where *B* has *r* columns, *A* has *r* rows, and *r* is much smaller than the dimensions of *W*~0~. Only "
           "*A* and *B* are trained, which reduces the number of trainable and communicated parameters from the "
           "size of *W*~0~ to *r* times the sum of its two dimensions. In the model studied here the adapted map "
           "acts on the pooled feature, and *W*~0~ is the identity (Section IV-B)."))
    add(("h2", "TRANSFER GAIN AND NEGATIVE TRANSFER"))
    add(p("To quantify the effect of federation on a participant, we define the transfer gain of strategy *m* for "
          "client *i* as its accuracy under *m* minus its accuracy under local-only training,"))
    add(("eq", r"\Delta_i(m) = \mathrm{Acc}_i(m) - \mathrm{Acc}_i(\mathrm{local})"))
    add(pc("Client *i* experiences negative transfer under *m* when this gain is negative "
           "\\cite{pan2010,wang2019neg}. The gain of a specialty is the mean over its three clients, and the gain "
           "of a federation is the mean over its 15 clients. A federation-level gain near zero is compatible with "
           "large gains and losses of opposite sign among its members, so Section VI reports the gains at all "
           "three levels."))
    add(("h2", "NOTATION"))
    add(p("{T:notation} lists the symbols used in the rest of the article."))
    add(("table", "notation"))

    # ============================== IV. METHODS
    add(("h1", "METHODS"))
    add(("h2", "DATA AND CLIENT CONSTRUCTION"))
    add(p(O("All images come from MedMNIST v2")))
    add(p("Preset A contains PathMNIST (colorectal cancer histology patches labeled with nine tissue types "
          "\\cite{kather2019}), DermaMNIST (dermatoscopic images of pigmented skin lesions in seven diagnostic "
          "categories \\cite{tschandl2018}), BloodMNIST (microscope images of eight types of peripheral blood cells "
          "\\cite{acevedo2020}), OCTMNIST (retinal optical coherence tomography scans in four diagnostic categories "
          "\\cite{kermany2018}), and TissueMNIST (kidney cortex cells in eight classes \\cite{ljosa2012}). Preset B "
          "contains OrganAMNIST and OrganCMNIST (axial and coronal views of abdominal computed tomography, each "
          "labeled with one of eleven organs \\cite{bilic2023}), PneumoniaMNIST (pediatric chest radiographs labeled "
          "as pneumonia or normal \\cite{kermany2018}), RetinaMNIST (fundus photographs graded on a five-level "
          "diabetic retinopathy scale \\cite{liu2022deepdrid}), and BreastMNIST (breast ultrasound images labeled "
          "as malignant or not \\cite{aldhabyani2020}). Preset B was chosen to contrast with Preset A: it mixes two "
          "large subsets with three small ones, whose clients have between 52 and 208 test images."))
    add(("table", "composition"))
    add(("h2", "MODEL"))
    add(p(O("The backbone is a ResNet-18")))
    add(("eq", E[0]))
    add(pc(O("where *A* is an *r* × *d* matrix")))
    add(p("{F:arch} shows the model and what is exchanged. The truncated backbone has about 2.8 million frozen "
          "parameters, so the adapter amounts to about 0.3% of the backbone, and a classifier for *C* classes has "
          "257*C* parameters."))
    add(("fig", "arch", "fig_arch.png",
         "Model and communication pattern. The backbone is frozen, the low-rank adapter is trained and exchanged "
         "with the server, and the classifier is trained and kept on the client."))
    add(("h2", "TRAINING STRATEGIES"))
    ext([p(O("All strategies used the Adam optimizer")), p(O("*Local-only:*")), p(O("*FedAvg*")), p(O("*FedProx*")),
         p(O("*FedSAA:*")), p(O("*Centralized:*"))])
    add(("h2", "FEDSAA AGGREGATION"))
    add(p(O("After local training in each round, the server holds")))
    add(("eq", E[1]))
    add(pc(O("and its cosine similarity")))
    add(("eq", E[2]))
    add(pc(O("where the inner product and norms are Frobenius")))
    add(("eq", E[3]))
    add(pc("which give a personalized aggregate,"))
    add(("eq", r"\bar{U}_i = \sum_{j=1}^{N} w_{ij}\,U_j"))
    add(pc("and, for comparison, an unweighted global mean,"))
    add(("eq", r"G = \frac{1}{N}\sum_{j=1}^{N} U_j"))
    add(pc(O("The two are blended")))
    add(("eq", E[5]))
    add(pc(O("and the blend is reduced to rank")))
    add(("eq", E[6]))
    add(pc(O("from which new factors")))
    add(("eq", E[7]))
    add(("eq", E[8]))
    add(pc(O("Here *ρ* is the retention coefficient")))
    add(p(O("FedSAA does not reduce to FedAvg") + " Algorithm 1 summarizes one run."))
    add(("alg", ALGORITHM))
    add(("h2", "COMPLEXITY AND COMMUNICATION COST"))
    add(p("Under every federated strategy each client uploads and downloads 2*rd* = 8,192 values per round, about "
          "32 kB in single precision. Over 40 rounds and 15 clients the total traffic is about 39 MB. FedAvg and "
          "FedProx need a number of server operations proportional to *Nrd* per round. FedSAA forms *N* products, "
          "*N*^2^ similarities, and *N* truncated SVDs of *d* × *d* matrices, at costs proportional to *Nrd*^2^, "
          "*N*^2^*d*^2^, and *Nd*^3^. With *d* = 256 and *N* = 15 this is small beside local training. Two "
          "differences matter more in practice: the server must keep *N* adapters instead of one, and each client "
          "receives a different adapter, so a new client cannot simply download a global model."))
    add(("h2", "EVALUATION"))
    add(p(O("After every round each client’s model was evaluated")))
    add(p("We also report the transfer gain defined in Section III-D at the client, specialty, and federation "
          "levels, and the standard deviation of accuracy across the 15 clients as a measure of how evenly "
          "performance is distributed."))
    add(("h2", "STATISTICAL ANALYSIS"))
    ext([p(O("The unit of analysis is the seed")), p(O("Normality of the differences cannot be assessed"))])
    add(("h2", "IMPLEMENTATION"))
    add(p(O("Experiments were implemented in PyTorch")))
    add(("table", "config"))

    # ============================== V. ANALYSIS
    add(("h1", "ANALYSIS OF THE MODEL AND THE AGGREGATION RULE"))
    add(p("Several properties of the model and of FedSAA follow from their definitions alone. They do not depend on "
          "the data, and they frame the empirical results of Section VI. The client index is dropped in this "
          "section where no confusion can arise."))
    add(("h2", "THE SHARED ADAPTER DOES NOT ENLARGE THE HYPOTHESIS CLASS"))
    add(p("Write *s* = *α*/*r* and *M* = *I* + *sBA* for the adapter. With classifier weights *W*~*i*~ and bias "
          "*b*~*i*~, client *i* computes"))
    add(("eq", r"f_i(x) = W_i\,(I + sBA)\,x + b_i"))
    add(p("*Proposition 1:* The set of functions that client *i* can represent with an adapter and a linear "
          "classifier equals the set of affine functions of *x*, that is, the set represented by a linear "
          "classifier alone. Moreover, for any fixed adapter with invertible *M*, the classifier alone still "
          "reaches every function in that set."))
    add(p("*Proof:* Every function of the form above is affine in *x*, with weight matrix *W*~*i*~*M*. Conversely, "
          "any affine function with weight matrix *W*′ is obtained with *B* = 0 and *W*~*i*~ = *W*′, which proves "
          "the first claim. For the second, if *M* is invertible, then *W*~*i*~ = *W*′*M*^−1^ gives the weight "
          "matrix *W*′."))
    add(p("The matrix *M* is invertible unless −1/*s* is an eigenvalue of *BA*, which does not occur for generic "
          "factors. Two consequences follow. First, local-only training with an adapter is a reparameterized linear "
          "probe on the frozen features. Second, no shared adapter can make a function reachable for a client that "
          "its own classifier could not reach, and no invertible shared adapter can make one unreachable. "
          "Differences between strategies therefore cannot come from representational capacity. They can come only "
          "from the optimization path: the number of steps, the parameterization, and, for federated strategies, "
          "the fact that aggregation moves the adapter under a classifier that was fitted to the previous adapter."))
    add(("h2", "AVERAGING FACTORS IS NOT AVERAGING UPDATES"))
    add(p("Let *p*~*i*~ be aggregation weights that sum to one, and let *Ā* and *B̄* be the weighted averages of "
          "the factors. FedAvg returns the adapter whose update is the product of the averaged factors. Expanding "
          "the products gives the identity"))
    add(("eq", r"\begin{aligned}\sum_{i} p_i B_i A_i - \bar{B}\bar{A} &= \sum_{i} p_i\,(B_i-\bar{B})(A_i-\bar{A})\end{aligned}"))
    add(p("*Proposition 2:* The update applied by FedAvg differs from the weighted mean of the client updates by "
          "the weighted cross-covariance of the factors, which is the right-hand side above."))
    add(p("The difference vanishes when all clients hold the same *A* or the same *B*, the situation that FFA-LoRA "
          "creates by freezing *A* \\cite{ffalora}, and it grows with the disagreement between clients. FLoRA "
          "refers to this term as aggregation noise \\cite{flora}. FedSAA avoids it by aggregating the products "
          "*U*~*i*~, at the cost of the truncation and refactorization analyzed below."))
    add(("h2", "HOW MUCH WEIGHT REACHES OTHER SPECIALTIES"))
    add(p("The similarity of an update to itself is 1, so the attention weight that client *i* gives to its own "
          "update is"))
    add(("eq", r"w_{ii} = \frac{e^{1/\tau}}{e^{1/\tau} + \sum_{j\neq i} e^{S_{ij}/\tau}}"))
    add(pc("The weights of the peers are bounded away from zero: every similarity is at least −1, so each peer "
           "receives at least a fraction *e*^−2/*τ*^ of the weight that the client gives to itself. To see what "
           "this implies in a federation shaped like ours, consider an idealized case in which the two peers of "
           "the same specialty have similarity *s* with client *i* and the 12 clients of other specialties have "
           "similarity 0. Then"))
    add(("eq", r"w_{ii} = \frac{e^{1/\tau}}{e^{1/\tau} + 2e^{s/\tau} + 12}"))
    add(pc("and the 12 other-specialty clients together receive 12 divided by the same denominator. After blending "
           "with the global mean, in which other specialties hold 12/15 of the weight, their share is (1 − *λ*) "
           "times that fraction plus 0.8*λ*."))
    add(p("{T:selfweight} and {F:selfweight} evaluate these expressions. At the default temperature of 0.5, other "
          "specialties receive 38–56% of the personalized aggregate and 59–68% of the blended update, even when "
          "same-specialty updates are highly similar (*s* = 0.9). The weighting isolates a client from other "
          "specialties only at low temperature. At *τ* = 0.1 the personalized aggregate is almost entirely the "
          "client’s own update unless same-specialty similarity is very high, and the global blend then supplies "
          "all of the cross-specialty weight (40% at *λ* = 0.5). These are illustrations under assumed "
          "similarities; the similarity matrices of the actual runs were not logged."))
    add(("table", "selfweight"))
    add(("fig", "selfweight", "fig_selfweight.png",
         "Share of the aggregated update that comes from the 12 clients of other specialties, as a function of the "
         "temperature, under the idealized similarity structure of the text. Blue: personalized aggregate. Orange: "
         "after blending with the global mean (λ = 0.5). Line styles give the similarity s of same-specialty peers."))
    add(("h2", "FACTOR-SPACE RETENTION DEPENDS ON AN ARBITRARY BASIS"))
    add(p("Let *A*′ and *B*′ be the factors obtained from the SVD, *A* and *B* the locally trained factors of the "
          "client, and *A*^+^ and *B*^+^ the factors after retention. Expanding the product gives"))
    add(("eq", r"\begin{aligned}B^{+}A^{+} &= (1-\rho)^2\,B'A' + \rho^2\,BA \\ &\quad + \rho(1-\rho)\,(B'A + BA')\end{aligned}"))
    add(pc("The first two terms are scaled copies of the aggregated and local updates. The last term multiplies "
           "factors that come from two unrelated factorizations. Subtracting the interpolation of the updates "
           "themselves isolates the distortion:"))
    add(("eq", r"\begin{aligned}B^{+}A^{+} &- \big[(1-\rho)\,B'A' + \rho\,BA\big] \\ &= -\rho(1-\rho)\,(B'-B)(A'-A)\end{aligned}"))
    add(p("*Proposition 3:* Factor-space retention equals interpolation of the updates only if the right-hand side "
          "above vanishes. That term depends on the factors and not only on their products, so it changes when "
          "either factorization is replaced by an equivalent one."))
    add(p("A factorization is unique only up to an invertible *r* × *r* matrix *R*, since (*B*′*R*)(*R*^−1^*A*′) = "
          "*B*′*A*′. The SVD fixes one such choice and local training arrives at another, and nothing aligns the "
          "two. A simple case shows the size of the effect. Suppose the aggregate equals the client’s own update, "
          "*B*′*A*′ = *BA*, but the SVD returns the factors with the opposite sign, *B*′ = −*B* and *A*′ = −*A*; "
          "the sign of singular vectors is arbitrary. Interpolating the updates would return *BA* unchanged. "
          "Factor-space retention returns"))
    add(("eq", r"B^{+}A^{+} = (1-2\rho)^2\,BA"))
    add(pc("which is 0.16*BA* at the default *ρ* = 0.3 and zero at *ρ* = 0.5. The outcome of retention therefore "
           "depends on a choice that carries no information. With *ρ* = 0 the step is inactive, and with *ρ* = 1 "
           "the client keeps its local adapter; the distortion is largest in between. A basis-independent "
           "alternative is to interpolate the updates, (1 − *ρ*)*Ũ*~*i*~ + *ρU*~*i*~, before a single truncated SVD."))
    add(("h2", "RANK TRUNCATION DISCARDS PART OF EVERY AGGREGATE"))
    add(p("The blended update of a client is a combination of *N* updates of rank at most *r*, so its rank can "
          "reach min(*d*, *Nr*) = 240 in our configuration. The truncated SVD returns the closest matrix of rank "
          "*r* in the Frobenius norm \\cite{eckart1936} and discards the remaining singular directions in every "
          "round. How much is lost depends on how concentrated the spectrum is; the loss is smallest when the "
          "aggregate is dominated by one client and largest when dissimilar updates are mixed evenly. The singular "
          "values were not logged in our runs, so the size of this loss is unknown."))

    # ============================== VI. RESULTS
    add(("h1", "RESULTS"))
    add(("h2", "FEDERATED TRAINING DID NOT OUTPERFORM LOCAL-ONLY TRAINING"))
    add(p(O("In Preset A the three federated strategies reached")))
    add(p(O("In Preset B the five strategies spanned") + " {F:seeds} shows every run, and {F:forest} shows the "
          "paired differences with their confidence intervals."))
    add(("wide_group", ["main"]))
    add(p(O("Centralized training was not a reliable upper bound")))
    add(("fig", "seeds", "fig_seeds.png",
         "Accuracy of every run at round 40. Dots are the three seeds and horizontal bars are their means."))
    add(("table", "contrasts"))
    add(("fig", "forest", "fig_forest.png",
         "Paired accuracy differences between strategies. Diamonds are means over three seeds, gray dots are the "
         "per-seed differences, and lines are 95% confidence intervals from the t distribution with two degrees of "
         "freedom."))
    add(("h2", "FEDSAA DID NOT IMPROVE ON FEDAVG"))
    ext([p(O("FedSAA was 0.15 points below FedAvg")), p(O("FedSAA varied least across seeds"))])
    add(("h2", "OTHER METRICS AND THE SPREAD ACROSS CLIENTS"))
    add(p(O("Macro-F1 was lower for every federated strategy")))
    add(p("The Appendix lists all pairwise comparisons for the four metrics ({T:pairs_acc} to {T:pairs_worst}). "
          "After Holm correction, eight of the 80 comparisons had an adjusted *p* value below 0.05, and none of "
          "them concerned accuracy or worst-client accuracy. Three concerned macro-F1 in Preset A: local-only "
          "versus FedSAA (0.020), centralized versus FedSAA (0.035), and centralized versus FedProx (0.035). Five "
          "concerned AUC: in Preset A, centralized versus FedSAA (0.019), local-only versus FedSAA (0.024), and "
          "centralized versus FedProx (0.031); in Preset B, local-only versus FedAvg (0.026) and local-only versus "
          "FedProx (0.035). In every one of these the federated strategy had the lower value."))
    add(p("Federation did not make accuracy more even across clients. The standard deviation of accuracy over the "
          "15 clients was 14.2–14.4 points for the federated strategies against 13.4 for local-only training in "
          "Preset A, and 16.5–18.0 against 15.2 in Preset B."))
    add(("h2", "THE EFFECT OF FEDERATION DEPENDS ON THE SPECIALTY"))
    add(p(O("Averaging over clients hides opposite effects")))
    add(("fig", "specialty", "fig1.png",
         "Accuracy difference from local-only training at round 40 for each specialty. Diamonds are means over "
         "three seeds and dots are individual seeds. The accuracy of a specialty is the mean over its three "
         "clients. The letter in parentheses gives the federation."))
    add(p(O("FedSAA stayed within one point of local-only")))
    add(p("{T:spec_acc} gives the accuracy of every specialty under every strategy, and {T:spec_delta} gives the "
          f"transfer gains with their confidence intervals. With three seeds the intervals are wide. They exclude "
          f"zero for {n_excl} of the 40 specialty–strategy pairs, without correction for multiplicity."))
    add(("wide_group", ["spec_acc", "spec_delta"]))
    add(("h2", "CLIENT-LEVEL EFFECTS"))
    add(p(O("At the client level, FedAvg exceeded local-only")))
    add(p("{F:clients} shows the transfer gain of every client. Under FedAvg the three clients of a specialty moved "
          "in the same direction in nine of the ten specialties; the exception was BreastMNIST, whose clients have "
          "52 test images each. Under FedSAA the gains were small and of mixed sign in eight specialties, and "
          "negative for all three clients in BloodMNIST and OCTMNIST. {T:clients} summarizes the distribution. The "
          "median client lost 0.8 points under FedAvg and 0.9 under FedSAA in Preset A. In Preset B the median "
          "client gained 1.4 points under FedAvg and lost 0.4 under FedSAA. The accuracy of every client is given "
          "in the Appendix ({T:client_full})."))
    add(("fig", "clients", "fig_clients.png",
         "Transfer gain of every client relative to local-only training, averaged over three seeds. Each specialty "
         "has three clients, shown as three rows."))
    add(("table", "clients"))
    add(("h2", "FEDERATED RUNS HAD NOT CONVERGED AT ROUND 40"))
    add(p(O("Local-only training came within one point")))
    add(("fig", "conv", "fig2.png",
         "Client-averaged test accuracy by communication round. Lines are means over three seeds and bands are "
         "±1 SD. One round is three local epochs."))
    add(p(O("The gap between local-only and FedAvg in Preset A")))
    add(p("{T:conv} gives the accuracy at four checkpoints and the slope over the last ten rounds. The slope of "
          "local-only training was slightly negative in both presets, that of centralized training was close to "
          "zero, and those of the federated strategies were positive, largest for FedSAA (0.15 and 0.19 points per "
          "round)."))
    add(("table", "conv"))
    add(("h2", "SENSITIVITY ANALYSIS POINTS TO THE RETENTION STEP"))
    add(p(O("The hyperparameters of FedSAA were varied") + " {F:ablation} plots the same values against the "
          "same-seed accuracies of FedAvg and local-only training."))
    add(p(O("The clearest signal came from the retention coefficient")))
    add(("table", "sensitivity"))
    add(("fig", "ablation", "fig_ablation.png",
         "One-at-a-time sensitivity of FedSAA in Preset A with seed 0. Horizontal lines are the accuracies of "
         "local-only training and FedAvg with the same seed. Each point is a single run."))
    add(p("These observations agree with the analysis of Section V. Rank cannot matter much if the model class is "
          "that of a linear probe (Proposition 1). The lowest temperature tested, 0.1, gave the best accuracy among "
          "the temperatures (68.3%), and it is the setting in which the personalized aggregate is closest to the "
          "client’s own update. Removing retention removes the basis-dependent distortion of Proposition 3. "
          "Because each setting was run once, these agreements are suggestive only."))
    add(("h2", "COMPUTATIONAL COST"))
    add(p("{T:runtime} gives the wall-clock time of a run. A federated run took 211–236 s in Preset A and 93–108 s "
          "in Preset B, about the same as local-only training (204 s and 92 s), because local training dominates "
          "the cost. FedSAA took 8% longer than FedAvg in Preset A and 15% longer in Preset B, and it was close to "
          "FedProx. The whole study of 30 runs took 73 minutes on one GPU."))
    add(("table", "runtime"))
    add(("h2", "PRECISION OF THE ESTIMATES"))
    ext([p(O("Two sources of noise limit these comparisons")), p(O("Second, test sets are finite"))])

    # ============================== VII. DISCUSSION
    add(("h1", "DISCUSSION"))
    add(("h2", "WHAT THE RESULTS SHOW"))
    ext([p(O("In two simulated cross-specialty federations")), p(O("The per-specialty results qualify")),
         p(O("We have no evidence for why"))])
    add(("h2", "WHY FEDSAA DID NOT HELP"))
    add(p("Three features of the design of FedSAA could explain its result, and Section V shows that each follows "
          "from the definition of the rule. First, the softmax weighting is diffuse. At the default temperature, "
          "between 59% and 68% of the blended update of a client comes from clients of other specialties under "
          "the idealized similarity structure of {T:selfweight}, even when same-specialty updates are highly "
          "similar. The rule therefore cannot isolate a client from dissimilar peers, which is consistent with "
          "FedSAA retaining the largest losses of FedAvg."))
    ext([p(O("Second, the blended update is truncated")), p(O("Third, the retention step interpolates"))])
    add(("h2", "THE MODEL CLASS LIMITS THE INTERPRETATION"))
    ext([p(O("The adapter is a linear map and the classifier is linear")), p(O("It also means that the study says nothing")),
         p(O("The two reference strategies are not bounds"))])
    add(("h2", "RELATION TO PRIOR WORK"))
    add(p("Our findings agree with several observations in the literature. Benchmarks of personalized FL report "
          "that the advantage of personalization varies across datasets and that no method wins everywhere "
          "\\cite{pflbench,motley}; FedSAA is one more personalized rule that did not beat plain averaging on a new "
          "benchmark. Work on local adaptation notes that a federated model can be worse than a locally trained "
          "one for individual participants \\cite{yu2020salvaging}, which is what we observe for half of the "
          "specialties. Studies of federated LoRA identify the mismatch between averaging factors and averaging "
          "products as a source of error \\cite{flora,ffalora,fedsalora}. Our analysis adds that correcting it by "
          "product-space aggregation introduces a truncation and, if the factors are then interpolated, a "
          "basis-dependent distortion."))
    add(p("Our study also differs from most of that literature in scope. Federated LoRA has mainly been studied "
          "for language models, with adapters inside nonlinear layers and with clients that share a task. Here the "
          "adapter is linear and placed after a frozen backbone, and the clients do not share a task. Recent "
          "methods for federations that span imaging modalities train much larger models, in one case a foundation "
          "model from scratch \\cite{omnifm,fm2}. Their setting and ours concern different model classes, and our "
          "results do not bear on theirs."))
    add(("h2", "THREATS TO VALIDITY"))
    add(p("*Internal validity:* The federated runs had not converged, so the ranking holds for one training budget "
          "only. One learning rate was used for all strategies without separate tuning, and the defaults of FedSAA "
          "were fixed in advance; the sensitivity analysis suggests they were not the best choice. Centralized "
          "training pooled classifier data across the three clients of a specialty, which the other strategies did "
          "not, so its comparison with them confounds two factors."))
    add(p("*Statistical conclusion validity:* Three seeds give wide intervals, and the *t*-based *p* values rest on "
          "a normality assumption that cannot be checked. Several clients have small test sets, and per-image "
          "predictions were not saved, so paired tests on images were not possible. The sensitivity analysis used "
          "one seed, and only its summary values were archived."))
    add(p("*Construct validity:* Accuracy averaged over clients weights a client with 52 test images as much as "
          "one with 1,000. Macro-F1 and AUC partly address class imbalance, but they were also averaged over "
          "clients with equal weight. The adapter is linear, so the construct tested is a shared linear "
          "reparameterization and not low-rank adaptation of a nonlinear network."))
    add(p("*External validity:* The clients are simulated by partitioning public datasets, so clients of the same "
          "specialty do not differ by scanner, protocol, or population as real hospitals would. Images were "
          "64 × 64 pixels, and training sets were capped at 2,000 images per client. We used one backbone and one "
          "adapter placement. We did not study privacy mechanisms, partial participation, or communication "
          "constraints. Nothing here constitutes clinical validation."))
    add(("h2", "PRACTICAL IMPLICATIONS AND RECOMMENDATIONS"))
    add(p(O("Three practical points follow")))
    add(p("For benchmarks of this kind we recommend the following."))
    for t in [
        "1) Report a local-only baseline for every participant, and report transfer gains per participant and per "
        "specialty, not only their mean.",
        "2) Train until the curves are flat, or report results at several budgets, and always show the curves.",
        "3) Use at least ten seeds, save per-image predictions, and test paired differences on images as well as "
        "on seeds.",
        "4) Log the similarity matrices and the singular values of the aggregated updates, so that the behavior of "
        "similarity-based and SVD-based rules can be inspected.",
        "5) When the trainable part is linear, include a linear probe, because the two share a hypothesis class.",
        "6) Release the raw per-run records and the scripts that produce every table and figure.",
    ]:
        add(p(t))

    # ============================== VIII. CONCLUSION, appendix, end matter
    add(("h1", "CONCLUSION"))
    add(p(O("We benchmarked five strategies")))
    add(("h1u", "APPENDIX"))
    add(("h1u", "COMPLETE COMPARISONS AND CLIENT-LEVEL RESULTS"))
    add(p("{T:pairs_acc} to {T:pairs_worst} list all ten pairwise comparisons in each preset for accuracy, "
          "macro-F1, AUC, and worst-client accuracy. {T:client_full} lists the accuracy of every client."))
    add(("wide_group", ["pairs_acc", "pairs_f1", "pairs_auc", "pairs_worst", "client_full"]))
    add(("h1u", "DATA AND CODE AVAILABILITY"))
    add(p(O("MedMNIST v2 is publicly available")))
    add(("h1u", "ACKNOWLEDGMENT"))
    add(p(O("[[AUTHORS: edit this statement")))
    return B, T
