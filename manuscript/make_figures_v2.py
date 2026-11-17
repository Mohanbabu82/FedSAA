#!/usr/bin/env python3
"""Figures for the revised IEEE Access manuscript, drawn from the Kaggle results.

Usage:
    python make_figures_v2.py <results_dir> <analysis_dir> <work_dir> <out_dir>

results_dir  : results/ from fedbench_v2_results.zip (runs.jsonl, meta_*.json)
analysis_dir : analysis/ from the same zip (T_similarity_energy.csv)
work_dir     : folder holding es.csv, es_pairs.csv and spec_gain_sel.csv (derived_tables.py)
Every plotted value is computed from those files.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

LABEL = {"local": "Local-only", "centralized": "Centralized", "fedavg": "FedAvg", "fedprox": "FedProx",
         "oracle": "Specialty oracle", "flexlora": "FlexLoRA", "ffa": "FFA-LoRA", "fedsa": "FedSA-LoRA",
         "fedsaa": "FedSAA", "fedsaa_r0": "FedSAA (ρ = 0)", "fedsaa_u": "FedSAA-U", "probe": "Linear probe"}
COLOR = {"local": "#0072B2", "centralized": "#000000", "fedavg": "#E69F00", "fedprox": "#CC79A7",
         "oracle": "#009E73", "flexlora": "#56B4E9", "ffa": "#999999", "fedsa": "#B8A000",
         "fedsaa": "#D55E00", "fedsaa_r0": "#8C2D04", "fedsaa_u": "#7B3294", "probe": "#666666"}
NICE = {"pathmnist": "PathMNIST", "dermamnist": "DermaMNIST", "bloodmnist": "BloodMNIST", "octmnist": "OCTMNIST",
        "tissuemnist": "TissueMNIST", "organamnist": "OrganAMNIST", "pneumoniamnist": "PneumoniaMNIST",
        "retinamnist": "RetinaMNIST", "breastmnist": "BreastMNIST", "organcmnist": "OrganCMNIST"}
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
W = 3.4


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=600)
    fig.savefig(os.path.join(out, name + ".pdf"))
    plt.close(fig)


def arch(out):
    fig, ax = plt.subplots(figsize=(W, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="0.25", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.4, linespacing=1.15)

    def arrow(x0, y0, x1, y1, style="-|>"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=style, lw=0.8, color="0.2"))

    ax.text(2.95, 9.75, "Client i (one hospital, one specialty)", ha="center", fontsize=6.8, fontweight="bold")
    box(0.3, 8.3, 5.3, 0.95, "Image, 64 × 64 × 3", "white")
    box(0.3, 6.55, 5.3, 1.25, "ResNet-18, stages 1–3\nfrozen, ImageNet-pretrained", "#D9D9D9")
    box(0.3, 4.55, 5.3, 1.5, "Low-rank adapter on feature x\nx + s·BAx  or  x + s·B·ReLU(Ax)\ntrained; A, B sent to server", "#FBD9A5")
    box(0.3, 2.55, 5.3, 1.5, "Linear classifier\ntrained; stays on the client", "#BBD7EE")
    box(0.3, 1.0, 5.3, 1.05, "Class scores", "white")
    for y0, y1 in [(8.3, 7.8), (6.55, 6.05), (4.55, 4.05), (2.55, 2.05)]:
        arrow(2.95, y0, 2.95, y1)
    box(6.9, 4.3, 2.9, 2.0, "Server\naggregates\nA, B or BA", "#FBD9A5")
    arrow(5.6, 5.6, 6.9, 5.6); arrow(6.9, 5.0, 5.6, 5.0)
    box(6.9, 1.7, 2.9, 1.6, "14 other\nclients", "white")
    arrow(8.35, 3.3, 8.35, 4.3, style="<|-|>")
    save(fig, out, "fig_arch")


def convergence(results, out):
    rows = [json.loads(l) for l in open(os.path.join(results, "runs.jsonl"))]
    rows = [r for r in rows if r["stage"] == "main" and r["adapter"] == "linear"]
    show = ["local", "centralized", "fedavg", "flexlora", "oracle", "fedsaa", "fedsaa_u"]
    fig, axes = plt.subplots(2, 1, figsize=(W, 4.3), sharex=True)
    for ax, p in zip(axes, "AB"):
        for s in show:
            cur = np.array([r["test_curve"] for r in rows if r["preset"] == p and r["strategy"] == s]) * 100
            x = np.arange(1, cur.shape[1] + 1)
            mu, sd = cur.mean(0), cur.std(0, ddof=1)
            ax.fill_between(x, mu - sd, mu + sd, color=COLOR[s], alpha=0.12, lw=0)
            ax.plot(x, mu, "--" if s == "centralized" else "-", color=COLOR[s], lw=1.0, label=LABEL[s])
        for t in (40, 100, 200):
            ax.axvline(t, color="0.75", lw=0.5, ls=":")
        ax.set_title(f"Preset {p}", loc="left", fontsize=7.5, fontweight="bold", pad=3)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_ylim(60 if p == "A" else 70, 76 if p == "A" else 82.5)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        ax.set_xlim(1, 200)
    axes[-1].set_xlabel("Communication round")
    axes[0].legend(frameon=False, fontsize=5.6, ncol=2, loc="lower right")
    fig.subplots_adjust(hspace=0.2)
    save(fig, out, "fig_conv")


def es_dots(work, out):
    es = pd.read_csv(os.path.join(work, "es.csv"))
    order = ["centralized", "oracle", "flexlora", "fedsaa_u", "fedsaa_r0", "fedprox", "fedsa", "fedsaa", "ffa", "fedavg", "probe", "local"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), sharey=True)
    for ax, p in zip(axes, "AB"):
        for k, s in enumerate(order):
            for a, dy, mk in (("linear", 0.15, "o"), ("nonlinear", -0.15, "s"), ("none", 0.0, "D")):
                v = es[(es.preset == p) & (es.adapter == a) & (es.strategy == s)].es_test.values
                if len(v) == 0:
                    continue
                m, h = v.mean(), 2.262 * v.std(ddof=1) / np.sqrt(len(v))
                ax.errorbar(m, -k + dy, xerr=h, fmt=mk, ms=2.8, lw=0.7, color=COLOR[s],
                            mfc=COLOR[s] if a != "nonlinear" else "white", mew=0.7)
        loc = es[(es.preset == p) & (es.adapter == "linear") & (es.strategy == "local")].es_test.mean()
        ax.axvline(loc, color=COLOR["local"], lw=0.6, ls=":")
        ax.set_title(f"Preset {p}", loc="left", fontsize=7.5, fontweight="bold", pad=3)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
        ax.set_xlabel("Test accuracy (%)")
    axes[0].set_yticks([-k for k in range(len(order))])
    axes[0].set_yticklabels([LABEL[s] for s in order])
    axes[1].plot([], [], "o", color="0.3", ms=3, label="linear adapter")
    axes[1].plot([], [], "s", color="0.3", mfc="white", ms=3, label="nonlinear adapter")
    axes[1].plot([], [], "D", color="0.3", ms=3, label="no adapter")
    fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, fontsize=5.5, ncol=3, loc="lower center", bbox_to_anchor=(0.55, -0.06))
    fig.subplots_adjust(wspace=0.08, bottom=0.17)
    save(fig, out, "fig_es")


def similarity(analysis, out):
    s = pd.read_csv(os.path.join(analysis, "T_similarity_energy.csv"))
    s = s[s.strategy.isin(["fedsaa", "fedsaa_u"])]
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.9), sharey=True)
    for ax, p in zip(axes, "AB"):
        for st, ls in (("fedsaa", "-"), ("fedsaa_u", "--")):
            g = s[(s.preset == p) & (s.strategy == st)].sort_values("round")
            ax.plot(g["round"], g.S_same, ls, marker="o", ms=2.5, color="#009E73", lw=0.9)
            ax.plot(g["round"], g.S_other, ls, marker="o", ms=2.5, color="#D55E00", lw=0.9)
            ax.plot(g["round"], g.w_other, ls, marker="s", ms=2.5, color="0.35", lw=0.9)
        ax.axhline(12 / 15, color="0.6", lw=0.5, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 40, 200]); ax.set_xticklabels(["1", "10", "40", "200"]); ax.minorticks_off()
        ax.set_title(f"Preset {p}", loc="left", fontsize=7.5, fontweight="bold", pad=3)
        ax.set_xlabel("Round")
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("Similarity or weight")
    h = [plt.Line2D([], [], color="#009E73", marker="o", ms=2.5, lw=0.9),
         plt.Line2D([], [], color="#D55E00", marker="o", ms=2.5, lw=0.9),
         plt.Line2D([], [], color="0.35", marker="s", ms=2.5, lw=0.9),
         plt.Line2D([], [], color="0.3", lw=0.9), plt.Line2D([], [], color="0.3", lw=0.9, ls="--")]
    fig.legend(h, ["similarity, same specialty", "similarity, other specialties", "weight on other specialties",
                   "FedSAA", "FedSAA-U"], frameon=False, fontsize=5.4, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.2))
    fig.subplots_adjust(wspace=0.08)
    save(fig, out, "fig_sim")


def specialty(work, out):
    g = pd.read_csv(os.path.join(work, "spec_gain_sel.csv"))
    g = g[(g.adapter == "linear") & g.strategy.isin(["fedavg", "flexlora", "oracle", "fedsaa_u"])]
    names = list(dict.fromkeys(g.preset + ":" + g.specialty))
    off = {"fedavg": 0.27, "flexlora": 0.09, "oracle": -0.09, "fedsaa_u": -0.27}
    fig, ax = plt.subplots(figsize=(W, 3.6))
    for s in off:
        sub = g[g.strategy == s]
        y = [-names.index(n) + off[s] for n in sub.preset + ":" + sub.specialty]
        ax.errorbar(sub.gain, y, xerr=[sub.gain - sub.lo, sub.hi - sub.gain], fmt="D", ms=2.8, lw=0.6,
                    color=COLOR[s], label=LABEL[s])
    for k in range(1, len(names)):
        if names[k].split(":")[0] != names[k - 1].split(":")[0]:
            ax.axhline(-k + 0.5, color="0.6", lw=0.5)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks([-i for i in range(len(names))])
    ax.set_yticklabels([NICE[n.split(":")[1]] + f" ({n.split(':')[0]})" for n in names])
    ax.set_xlabel("Gain over local-only (percentage points), mean and 95% CI")
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    ax.legend(frameon=False, ncol=4, fontsize=5.6, loc="lower center", bbox_to_anchor=(0.4, 1.0),
              handletextpad=0.2, columnspacing=0.8)
    save(fig, out, "fig_spec")


if __name__ == "__main__":
    results, analysis, work, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    arch(out)
    convergence(results, out)
    es_dots(work, out)
    similarity(analysis, out)
    specialty(work, out)
    print("figures written to", out)
