"""Regenerates fig:asym and fig:mech for the RECOMB (idea3) paper from the exact
numbers used in Table 2 and Table 4 of main.tex. The inherited figures from
PAPER_C_coldstart (figC1_asymmetry, figC3_mechanism) were generated under a
different convention than the paper text: figC1 plots corrected-pooling
expression-view deltas (not the naive-pooling numbers in Table 2) and its
significance stars come from an unpaired per-signature test rather than the
seed-paired Wilcoxon test the Methods section describes; figC3 plots the
concatenated-view mechanism test, not the structure-view test in Table 4.
Both are internally consistent within themselves but inconsistent with this
paper's own text, so they are regenerated here from results/coldstart_significance.csv
(naive pooling) and results/mechanism_test_structure.csv.

Writes figures/fig_asymmetry.pdf and figures/fig_mechanism.pdf.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = f"{ROOT}/results"
FIGS = f"{ROOT}/figures"

ORDER = ["Antipsychotic", "AED", "SSRI", "NSAID", "BetaBlocker", "Statin"]
LABELS = {"Antipsychotic": "Antipsych.", "AED": "AED", "SSRI": "SSRI",
          "NSAID": "NSAID", "BetaBlocker": "B-blocker", "Statin": "Statin"}


def stars(p):
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fig_asymmetry():
    df = pd.read_csv(f"{RESULTS}/coldstart_significance.csv")
    df["cls"] = df["class"].str.replace("class_", "", regex=False)
    struct = df[df.view == "structure"].set_index("cls")
    expr = df[df.view == "expression"].set_index("cls")

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = range(len(ORDER))
    w = 0.38
    bs = [struct.loc[c, "classholdout_minus_full"] for c in ORDER]
    be = [expr.loc[c, "classholdout_minus_full"] for c in ORDER]
    ps = [struct.loc[c, "wilcoxon_p"] for c in ORDER]
    pe = [expr.loc[c, "wilcoxon_p"] for c in ORDER]

    ax.bar([i - w/2 for i in x], bs, width=w, label="structure view", color="#1f77b4")
    ax.bar([i + w/2 for i in x], be, width=w, label="expression view", color="#d95f02")
    for i, (v, p) in enumerate(zip(bs, ps)):
        s = stars(p)
        if s:
            ax.text(i - w/2, v + (0.012 if v >= 0 else -0.022), s, ha="center", fontsize=11)
    for i, (v, p) in enumerate(zip(be, pe)):
        s = stars(p)
        if s:
            ax.text(i + w/2, v + (0.012 if v >= 0 else -0.022), s, ha="center", fontsize=11)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[c] for c in ORDER])
    ax.set_ylabel("class-holdout minus full\n(AUROC)")
    ax.legend(loc="lower left", frameon=False)
    ax.text(0.98, 0.95, "* p<0.05   ** p<0.01", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="gray")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig_asymmetry.pdf")
    fig.savefig(f"{FIGS}/fig_asymmetry.png", dpi=150)
    print("wrote fig_asymmetry.pdf/.png")


def fig_mechanism():
    df = pd.read_csv(f"{RESULTS}/mechanism_test_structure.csv")
    df["cls"] = df["cls"].str.replace("class_", "", regex=False)
    df = df.set_index("cls")

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = range(len(ORDER))
    w = 0.26
    full = [df.loc[c, "full"] for c in ORDER]
    deb = [df.loc[c, "debiased"] for c in ORDER]
    cold = [df.loc[c, "classholdout"] for c in ORDER]

    ax.bar([i - w for i in x], full, width=w, label="full", color="#1f77b4")
    ax.bar(list(x), deb, width=w, label="same-class pairs excluded\nfrom negatives", color="#d95f02")
    ax.bar([i + w for i in x], cold, width=w, label="class holdout", color="#2ca02c")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[c] for c in ORDER])
    ax.set_ylabel("AUROC (structure view)")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig_mechanism.pdf")
    fig.savefig(f"{FIGS}/fig_mechanism.png", dpi=150)
    print("wrote fig_mechanism.pdf/.png")


if __name__ == "__main__":
    fig_asymmetry()
    fig_mechanism()
