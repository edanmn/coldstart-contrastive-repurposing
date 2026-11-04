"""Compare each contrastive structure/concat view against raw ECFP4+RF on the
same subsampling splits (mean over pretraining seeds, Wilcoxon paired by split).
Reads results/main_results.csv. Writes results/views_vs_ecfp4rf.csv."""
import os
import pandas as pd
from scipy.stats import wilcoxon

R = os.path.join(os.path.dirname(__file__), "..", "results")
d = pd.read_csv(f"{R}/main_results.csv")
m = d.groupby(["class", "method", "seed"]).auroc.mean().reset_index()
rows = []
for c in m["class"].unique():
    s = m[m["class"] == c]
    rf = s[s.method == "raw_ecfp4_RF"].set_index("seed").auroc
    for meth in ["contrastive_full_structure", "contrastive_full_concat",
                 "contrastive_classholdout_structure", "contrastive_classholdout_concat",
                 "contrastive_debiased_structure"]:
        x = s[s.method == meth].set_index("seed").auroc.reindex(rf.index)
        rows.append(dict(cls=c.replace("class_", ""), method=meth, auroc=x.mean(),
                         ecfp4_rf=rf.mean(), diff=x.mean() - rf.mean(),
                         p=wilcoxon(x, rf).pvalue))
out = pd.DataFrame(rows)
out.to_csv(f"{R}/views_vs_ecfp4rf.csv", index=False)
print(out.to_string(index=False, float_format="%.4f"))
