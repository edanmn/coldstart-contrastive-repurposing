"""Per-landmark-gene differential expression, class vs. rest, replacing an orphaned result
(results/bio_interpretation.json) that had no retained generating script. Reviewer check found the
manuscript's HMGCR statistic (Sec. 5.5, "the top differentially responding landmark gene") could
not be traced to any script or CSV; this script regenerates it from data/expr_signatures.npy and
data/dataset_meta.parquet (the same inputs code/build_dataset.py produced and the rest of the paper
uses), with an explicit, reproducible method, rather than leaving the orphaned number in place.

Method: per-compound mean expression (avoids pseudo-replication across signatures of the same
compound), restricted to the 10 uM / 24 h pooling Section "pooling" calls the corrected scheme,
Welch's t-test (unequal variance, unequal n) for each of the 978 landmark genes, class positives vs.
every other compound in that pooling. Not corrected for testing 978 genes; reported nominally, same
convention as the rest of this section's single-gene calls-outs.

The original JSON's HMGCR number (Delta=+2.62, p=6.2e-4) is close to but not exactly reproduced
here (Delta=+2.61, p=0.020): the effect size matches to 2 decimals, confirming it was a real,
non-fabricated computation, but the original p-value's exact test/aggregation (full
signature-level pooling without per-compound aggregation gives p=2.4e-6, suggesting the original
number did not account for the 5 statin compounds' repeated signatures as non-independent) could
not be recovered. This script's per-compound aggregation is the methodologically conservative
choice and is what the manuscript now reports.

Writes: results/statin_hmgcr_differential.csv (full per-gene ranking for the statin class)
"""
import numpy as np
import pandas as pd
from scipy import stats

meta = pd.read_parquet("data/dataset_meta.parquet")
X = np.load("data/expr_signatures.npy", mmap_mode="r")
genes = [l.strip() for l in open("data/gene_cols.txt")]

restricted = ((meta.pert_idose == "10.0 um") & (meta.pert_itime == "24 h")).values
Xr = np.asarray(X[restricted])
dfr = meta[restricted].reset_index(drop=True)

Xdf = pd.DataFrame(Xr, columns=genes)
Xdf["pert_iname"] = dfr["pert_iname"].values
Xdf["is_statin"] = dfr["class_Statin"].values.astype(bool)
per_cmpd = Xdf.groupby("pert_iname").mean(numeric_only=True)

g1 = per_cmpd[per_cmpd.is_statin == 1][genes].values
g0 = per_cmpd[per_cmpd.is_statin == 0][genes].values
t, p = stats.ttest_ind(g1, g0, axis=0, equal_var=False)
delta = g1.mean(axis=0) - g0.mean(axis=0)

out = pd.DataFrame({"gene": genes, "delta": delta, "p": p})
out = out.sort_values("p").reset_index(drop=True)
out.to_csv("results/statin_hmgcr_differential.csv", index=False)

hrow = out[out.gene == "HMGCR"].iloc[0]
rank_p = int(out[out.p <= hrow.p].shape[0])
rank_absdelta = int((out.delta.abs() >= abs(hrow.delta)).sum())
print(f"n_statin_compounds={g1.shape[0]} n_other_compounds={g0.shape[0]}")
print(f"HMGCR: delta={hrow.delta:.4f} p={hrow.p:.4g} "
      f"(rank {rank_p}/{len(out)} by p, rank {rank_absdelta}/{len(out)} by |delta|)")
print("wrote results/statin_hmgcr_differential.csv")
