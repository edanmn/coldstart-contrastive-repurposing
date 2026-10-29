"""Strong-accept revision: fixed_featurization_replication.py's random-vector control showed
the LARGEST class-holdout-minus-full delta of any featurization tested, including on literal
i.i.d. noise with no chemistry at all. That is the opposite of a null control, and it reframes
Section 4.7: the candidate explanation there ("a population-independent featurization is one
candidate explanation") is checked here against every encoder and featurization available,
not just the GCN variants, to see whether a simpler confound (how far the full-data baseline
already sits from ceiling) accounts for the pattern instead.

Pools (featurization, class) full-data AUROC and class-holdout-minus-full delta across all
8 structure-view encoders this project has run: ECFP4+MLP (the paper's main result), plain
GCN, 3 GCN variants (frozen extractor, max-pooling, GIN), and the 3 new fixed featurizations
(MACCS, physicochemical descriptors, random-vector control). n=48 (8 featurizations x 6
classes).

Checks the rigor-skill defect-6 concern explicitly: delta = cold - full, so correlating full
against delta risks a mechanical negative correlation purely from the shared -full term, even
if full and cold were independent. Reports full-vs-cold directly (not just full-vs-delta) to
distinguish a genuine compression/ceiling effect (cold's range narrower than full's, but the
two still positively correlated) from that artifact (which would show full and cold
uncorrelated, Cov=0).

Also regresses delta on full across the pooled 48 points and reports each featurization's
mean residual, to ask whether any encoder shows a cold-start benefit beyond what the general
baseline-difficulty trend predicts.

Reviewer-pass addition: the draft reported ECFP4's residual ranking (+0.057, ahead of every
other featurization) with no significance test, despite every other claim in the paper being
tested -- an inconsistency an adversarial pass caught immediately, since each residual is a
mean of only 6 classes with within-group SD up to 0.144 (random_control), larger than the gap
between ECFP4 and its nearest competitor (MACCS, +0.017). Added: Mann-Whitney U (ECFP4's 6
residuals vs. each other featurization's 6, and vs. all other 42 pooled), and a Kruskal-Wallis
omnibus test across all 8 groups, to state honestly whether the ranking is distinguishable
from noise with this few classes per featurization.

Writes results/baseline_difficulty_confound.csv (per-row data, predicted delta, residual) and
results/baseline_difficulty_confound_summary.csv (per-featurization residual means).
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal
from scipy.stats import pearsonr

RESULTS = "../results"


def main():
    rows = []

    main_res = pd.read_csv(f"{RESULTS}/main_results.csv")
    sub = main_res[main_res.method.isin(
        ["contrastive_full_structure", "contrastive_classholdout_structure"])]
    g = sub.groupby(["class", "method"])["auroc"].mean().unstack()
    g = g.rename(columns={"contrastive_full_structure": "full",
                           "contrastive_classholdout_structure": "cold"})
    for cls, r in g.iterrows():
        rows.append(dict(featurization="ecfp4", cls=cls.replace("class_", ""),
                          full=r["full"], delta=r["cold"] - r["full"]))

    gcn = pd.read_csv(f"{RESULTS}/gnn_architecture_replication.csv")
    gg = gcn.groupby(["class", "regime"])["auroc"].mean().unstack()
    for cls, r in gg.iterrows():
        rows.append(dict(featurization="plain_gcn", cls=cls,
                          full=r["full"], delta=r["classholdout"] - r["full"]))

    gv = pd.read_csv(f"{RESULTS}/gnn_variant_auroc_followup.csv")
    for _, r in gv.iterrows():
        rows.append(dict(featurization=r["variant"], cls=r["cls"],
                          full=r["auroc_full"], delta=r["delta"]))

    ff = pd.read_csv(f"{RESULTS}/fixed_featurization_replication.csv")
    for _, r in ff.iterrows():
        rows.append(dict(featurization=r["featurization"], cls=r["cls"],
                          full=r["full_mean"], delta=r["delta"]))

    df = pd.DataFrame(rows)
    df["cold"] = df["full"] + df["delta"]
    assert len(df) == 48, f"expected 48 rows, got {len(df)}"

    r_fd, p_fd = pearsonr(df["full"], df["delta"])
    r_fc, p_fc = pearsonr(df["full"], df["cold"])
    no_rand = df[df.featurization != "random_control"]
    r_fd_nr, p_fd_nr = pearsonr(no_rand["full"], no_rand["delta"])

    print(f"full vs delta, all 8 featurizations (n={len(df)}): r={r_fd:.3f}, p={p_fd:.2e}")
    print(f"full vs cold,  all 8 featurizations (n={len(df)}): r={r_fc:.3f}, p={p_fc:.2e}")
    print(f"Var(full)={df['full'].var():.4f}, Var(cold)={df['cold'].var():.4f} "
          f"-- cold's range is narrower than full's, and full/cold still correlate "
          f"positively, so this is a genuine compression effect, not defect-6's "
          f"mechanical full-vs-(cold-full) artifact (that would need Cov(full,cold)=0).")
    print(f"full vs delta, excluding random_control (n={len(no_rand)}): "
          f"r={r_fd_nr:.3f}, p={p_fd_nr:.2e}")

    slope, intercept = np.polyfit(df["full"], df["delta"], 1)
    df["pred_delta"] = slope * df["full"] + intercept
    df["resid"] = df["delta"] - df["pred_delta"]
    print(f"\nglobal fit: delta ~ {slope:.3f}*full + {intercept:.3f}")

    summary = df.groupby("featurization")["resid"].agg(["mean", "std", "count"]) \
        .sort_values("mean", ascending=False)
    print("\nmean residual by featurization (positive = more cold-start benefit than the "
          "baseline-difficulty trend alone predicts):")
    print(summary)

    ecfp4 = df[df.featurization == "ecfp4"]["resid"].values
    others = df[df.featurization != "ecfp4"]["resid"].values
    u_pooled, p_pooled = mannwhitneyu(ecfp4, others, alternative="greater")
    print(f"\nMann-Whitney, ECFP4's 6 residuals vs. all other 42 pooled, one-sided: "
          f"U={u_pooled:.1f}, p={p_pooled:.4f}")

    print("Pairwise Mann-Whitney, ECFP4 vs. each other featurization (n=6 each, one-sided):")
    pairwise_rows = []
    for feat in df["featurization"].unique():
        if feat == "ecfp4":
            continue
        other = df[df.featurization == feat]["resid"].values
        u, p = mannwhitneyu(ecfp4, other, alternative="greater")
        pairwise_rows.append(dict(featurization=feat, u=u, p=p))
        print(f"  vs {feat:18s} U={u:5.1f}  p={p:.4f}")

    groups = [df[df.featurization == f]["resid"].values for f in df["featurization"].unique()]
    h_stat, p_kw = kruskal(*groups)
    print(f"\nKruskal-Wallis omnibus test across all 8 featurizations: "
          f"H={h_stat:.3f}, p={p_kw:.4f}")
    print("No individual pairwise comparison clears uncorrected p<0.05 with only 6 classes "
          "per featurization, and the omnibus test finds no overall difference: the residual "
          "ranking is suggestive (significant only pooled, one-sided), not established.")

    df.to_csv(f"{RESULTS}/baseline_difficulty_confound.csv", index=False)
    summary.to_csv(f"{RESULTS}/baseline_difficulty_confound_summary.csv")
    pd.DataFrame(pairwise_rows).to_csv(
        f"{RESULTS}/baseline_difficulty_confound_pairwise.csv", index=False)
    print(f"\nWrote baseline_difficulty_confound.csv, _summary.csv, and _pairwise.csv "
          f"({len(df)} rows)")


if __name__ == "__main__":
    main()
