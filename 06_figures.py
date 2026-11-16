"""Step 6: the paper's figures, drawn from the saved results (IEEE single-column width, 3.5 in).

Fig. 1  where SMOTE is applied (schematic)                 no data needed
Fig. 2  accuracy under P0-P3                               <out>/protocols_summary.json
Fig. 3  pooled confusion matrices, P0 and P2               <out>/confusion_pooled.json
Fig. 4  learning, ROC and precision-recall curves, fold 0  <out>/curves_fold0.json
Fig. 5  leak-free 5-class accuracy with 95% CIs            <out>/stats_vs_chance.csv
Output: <out>/figures/fig1-fig5 (.png at 600 dpi and .pdf).
"""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import args

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
W = 3.5
ORANGE, GREEN, BLUE, GREY, RED = "#D85A30", "#1D9E75", "#378ADD", "#888780", "#B03A2E"
PAPER = 92.39


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig1(out):
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.75))

    def box(ax, x, y, w, h, text, fc, ec="black"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.03", fc=fc, ec=ec, lw=0.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.2)

    def arrow(ax, x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.6, color="black"))
    for ax, title, leaky in ((axes[0], "(a) Published code: SMOTE, then split", True),
                             (axes[1], "(b) Correct: split, then SMOTE", False)):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_title(title, fontsize=6.6, pad=2)
        box(ax, 0.2, 0.84, 0.6, 0.12, "1,526 real recordings", "#F1EFE8"); arrow(ax, 0.5, 0.84, 0.5, 0.74)
        if leaky:
            box(ax, 0.1, 0.56, 0.8, 0.18, "SMOTE on all data\n4,905 samples (68.9% synthetic)", "#F9ECE7", RED)
            arrow(ax, 0.5, 0.56, 0.5, 0.46); box(ax, 0.1, 0.30, 0.8, 0.16, "6-fold split", "#F1EFE8")
            arrow(ax, 0.3, 0.30, 0.25, 0.20); arrow(ax, 0.7, 0.30, 0.75, 0.20)
            box(ax, 0.02, 0.02, 0.45, 0.18, "train:\nreal + synthetic", "#E5F4E4")
            box(ax, 0.53, 0.02, 0.45, 0.18, "test: mostly\nsynthetic", "#F9ECE7", RED)
        else:
            box(ax, 0.1, 0.56, 0.8, 0.18, "6-fold split\n(random or subject-wise)", "#F1EFE8")
            arrow(ax, 0.3, 0.56, 0.25, 0.46); arrow(ax, 0.7, 0.56, 0.75, 0.46)
            box(ax, 0.02, 0.28, 0.45, 0.18, "train fold", "#E5F4E4")
            box(ax, 0.53, 0.28, 0.45, 0.18, "test fold:\nreal only", "#E7F1FB", BLUE)
            arrow(ax, 0.245, 0.28, 0.245, 0.20); box(ax, 0.01, 0.02, 0.6, 0.18, "SMOTE and feature\nselection: train only", "#E5F4E4")
    plt.tight_layout(pad=0.2, w_pad=0.6); save(fig, out, "fig1_smote")


def fig2(res, out):
    s = json.load(open(f"{res}/protocols_summary.json"))
    get = lambda p, m: 100 * s[f"{p}/{m}"]["accuracy_mean"] if f"{p}/{m}" in s else np.nan
    protos = ["P0", "P1", "P2", "P3"]
    labels = ["P0\npublished", "P1\n+ separate\nvalidation", "P2\n+ SMOTE inside\ntraining folds", "P3\n+ subject-\nwise folds"]
    tn = [get(p, "TransNN-MHA") for p in protos]
    fig, ax = plt.subplots(figsize=(W, 1.9)); x = np.arange(4); w = 0.26
    ax.bar(x - w, tn, w, color=ORANGE, label="TransNN-MHA")
    ax.bar(x, [get(p, "KNN") for p in protos], w, color=GREY, label="KNN")
    ax.bar(x + w, [get(p, "Logistic regression") for p in protos], w, color="#B4B2A9", label="Logistic regression")
    for xi, v in zip(x - w, tn):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=6.3)
    ax.axhline(PAPER, ls=":", c=RED, lw=0.8); ax.text(3.45, PAPER + 1.1, f"paper: {PAPER}%", ha="right", fontsize=6.3, color=RED)
    ax.axhline(20, ls="--", c="black", lw=0.6); ax.text(1.0, 15.0, "chance 20%", ha="center", fontsize=6.3)
    ax.annotate("", xy=(1.78, tn[2] + 3), xytext=(1.2, tn[1] - 2), arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.7, mutation_scale=6))
    ax.text(1.62, 62, f"−{tn[1] - tn[2]:.1f} pts", color=RED, fontsize=6.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.1); ax.set_ylim(0, 105); ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=6, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.84)); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(pad=0.3); save(fig, out, "fig2_protocols")


def fig3(res, out):
    d = json.load(open(f"{res}/confusion_pooled.json"))
    order = ["Rest", "Real Right Fist", "Imaginary Right Fist", "Real Both Feet", "Imaginary Both Feet"]
    names = ["Rest", "Real fist", "Imag. fist", "Real feet", "Imag. feet"]
    pos = [d["classes"].index(c) for c in order]
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.95))
    for ax, p in ((axes[0], "P0"), (axes[1], "P2")):
        C = np.array(d["pooled"][f"{p}/TransNN-MHA"], float)[np.ix_(pos, pos)]; R = 100 * C / C.sum(1, keepdims=True)
        ax.imshow(R, cmap="Blues", vmin=0, vmax=100)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{R[i, j]:.0f}", ha="center", va="center", fontsize=5.6, color="white" if R[i, j] > 55 else "black")
        ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.tick_params(length=0)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=5.6); ax.set_yticklabels(names if p == "P0" else [""] * 5, fontsize=5.6)
        title = "(a) Published code (P0)" if p == "P0" else "(b) Leak-free (P2)"
        ax.set_title(f"{title}: {100 * np.trace(C) / C.sum():.1f}%", fontsize=6.4, pad=3); ax.set_xlabel("predicted", fontsize=5.8, labelpad=1)
    axes[0].set_ylabel("true", fontsize=5.8, labelpad=1)
    plt.tight_layout(pad=0.25, w_pad=0.4); save(fig, out, "fig3_confusion")


def fig4(res, out):
    d = json.load(open(f"{res}/curves_fold0.json"))
    C = {"P0 train": "#1f77b4", "P0 val": "#ff7f0e", "P2 train": "#2ca02c", "P2 val": "#d62728"}
    fig, ax = plt.subplots(2, 2, figsize=(W, 3.12))
    for tag in ("P0", "P2"):
        h = d[tag]["history"]; ep = np.arange(1, len(h["loss"]) + 1)
        ax[0, 0].plot(ep, h["loss"], color=C[f"{tag} train"], lw=1, label=f"{tag} train")
        ax[0, 0].plot(ep, h["val_loss"], color=C[f"{tag} val"], lw=1, ls="--", label=f"{tag} early stop")
        ax[0, 1].plot(ep, h["accuracy"], color=C[f"{tag} train"], lw=1)
        ax[0, 1].plot(ep, h["val_accuracy"], color=C[f"{tag} val"], lw=1, ls="--")
        col = C[f"{tag} train"]
        ax[1, 0].plot(d[tag]["roc_micro"]["fpr"], d[tag]["roc_micro"]["tpr"], color=col, lw=1, label=f"{tag} (AUC = {d[tag]['roc_micro']['auc']:.3f})")
        ax[1, 1].plot(d[tag]["pr_micro"]["recall"], d[tag]["pr_micro"]["precision"], color=col, lw=1, label=f"{tag} (AUC = {d[tag]['pr_micro']['auc']:.3f})")
    ax[1, 0].plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7); ax[1, 1].axhline(0.2, ls=":", color="gray", lw=0.7)
    for a_, (t, xl, yl) in zip(ax.ravel(), [("(a) Loss", "Epoch", "Loss"), ("(b) Accuracy", "Epoch", "Accuracy"),
                                            ("(c) ROC curve (micro-average)", "False positive rate", "True positive rate"),
                                            ("(d) Precision–recall curve (micro-average)", "Recall", "Precision")]):
        a_.set_title(t, pad=2); a_.set_xlabel(xl, labelpad=1); a_.set_ylabel(yl, labelpad=1)
        a_.tick_params(length=2, pad=1.5); a_.spines[["top", "right"]].set_visible(False)
    ax[0, 1].set_ylim(0, 1.05)
    for a_ in (ax[1, 0], ax[1, 1]):
        a_.set_xlim(0, 1); a_.set_ylim(0, 1.02)
    ax[1, 0].legend(frameon=False, loc="lower right", handlelength=1.4); ax[1, 1].legend(frameon=False, loc="center right", handlelength=1.4)
    plt.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.8, rect=(0, 0, 1, 0.945))
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, handlelength=1.6, columnspacing=0.9, handletextpad=0.4, bbox_to_anchor=(0.5, 1.0))
    save(fig, out, "fig4_curves")


def fig5(res, out):
    v = pd.read_csv(f"{res}/stats_vs_chance.csv")
    pick = [("TransNN-MHA, P3 (subject-wise)", "TransNN-MHA", "P3", ORANGE), ("TransNN-MHA, P2", "TransNN-MHA", "P2", ORANGE),
            ("KNN, P3", "KNN", "P3", GREY), ("Logistic regression, P3", "Logistic regression", "P3", GREY),
            ("CSP + LDA, 1-s trials", "CSP + LDA", "1s", GREEN), ("CSP + LDA, 4-s trials", "CSP + LDA", "4s", GREEN),
            ("EEGNet, 1-s trials", "EEGNet seed 42", "1s", BLUE), ("EEGNet, 4-s trials", "EEGNet seed 42", "4s", BLUE)]
    fig, ax = plt.subplots(figsize=(W, 1.95))
    for i, (lab, model, setting, c) in enumerate(pick):
        r = v[(v.task == "5class") & (v.model == model) & (v.setting == setting)].iloc[0]
        m, lo, hi = 100 * r.accuracy, 100 * r.ci_folds_low, 100 * r.ci_folds_high
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=c, ms=3.2, capsize=2, lw=0.8)
        ax.text(hi + 1.5, i, f"{m:.1f}", va="center", fontsize=6.2)
    ax.axvline(20, ls="--", c="black", lw=0.6); ax.text(20.8, -0.85, "chance", fontsize=6.2)
    ax.axvline(PAPER, ls=":", c=RED, lw=0.8); ax.text(PAPER - 0.9, -0.85, f"paper {PAPER}%", fontsize=6.2, color=RED, ha="right")
    ax.set_yticks(range(len(pick))); ax.set_yticklabels([p[0] for p in pick], fontsize=6.3); ax.set_ylim(len(pick) - 0.5, -1.3)
    ax.set_xlim(10, 100); ax.set_xlabel("5-class accuracy (%), 95% CI over folds"); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(pad=0.3); save(fig, out, "fig5_leak_free")


if __name__ == "__main__":
    a = args(__doc__.splitlines()[0])
    out = f"{a.out}/figures"; os.makedirs(out, exist_ok=True)
    fig1(out)
    for name, fn, need in (("Fig. 2", fig2, "protocols_summary.json"), ("Fig. 3", fig3, "confusion_pooled.json"),
                           ("Fig. 4", fig4, "curves_fold0.json"), ("Fig. 5", fig5, "stats_vs_chance.csv")):
        if os.path.exists(f"{a.out}/{need}"):
            fn(a.out, out); print(name, "saved")
        else:
            print(name, "skipped: run the script that writes", need)
