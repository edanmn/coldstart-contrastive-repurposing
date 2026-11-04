"""Cross-encoder summary of the membership control (Table tab:encoders). Reads
membership_v2_*_summary.csv, membership_matched_random_control_summary.csv (v1, random p2)
and lopo_matched_*.csv. Writes results/encoder_summary.csv."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from run_experiments import RESULTS
R = RESULTS
rows = []
specs = [("ECFP4 MLP", "ecfp4_p2", "P2", 3), ("ChemBERTa MLP", "chemberta_p2", "P2", 3),
         ("GCN", "gcn_p2", "P2", 3), ("D-MPNN (Finlayson et al.)", "finlayson_p2", "P2", 3),
         ("Random vectors", None, "P2", 3), ("ECFP4 MLP", "ecfp4_p1", "P1", 3),
         ("Random vectors", "random_p1", "P1", 3)]
for name, tag, ds, ns in specs:
    if tag is None:
        s = pd.read_csv(f"{R}/membership_matched_random_control_summary.csv").rename(
            columns={"holdout_minus_full": "hold_minus_full", "p_holdout_vs_full": "p_hold_vs_full",
                     "membership_auroc_matched": "mem_matched"})
        lopo = None
    else:
        s = pd.read_csv(f"{R}/membership_v2_{tag}_summary.csv")
        lopo = pd.read_csv(f"{R}/lopo_matched_{tag}.csv")
    rec = dict(encoder=name, dataset=ds, seeds=ns,
               mem_min=s.mem_matched.min(), mem_max=s.mem_matched.max(),
               full_min=s.full.min(), full_max=s.full.max(),
               n_hold_gain=int(((s.hold_minus_full > 0) & (s.p_hold_vs_full < 0.05)).sum()),
               n_matched_gain=int(((s.matched_minus_full > 0) & (s.p_matched_vs_full < 0.05)).sum()),
               n_matched_gain_bonf=int(((s.matched_minus_full > 0) & (s.p_matched_vs_full < 0.05 / 6)).sum()),
               n_matched_loss=int(((s.matched_minus_full < 0) & (s.p_matched_vs_full < 0.05)).sum()))
    if lopo is not None:
        dm = (lopo.matched - lopo.full).values
        bs = [np.random.RandomState(i).choice(dm, len(dm)).mean() for i in range(5000)]
        rec.update(lopo_n=len(dm), lopo_delta=dm.mean(), lopo_lo=np.percentile(bs, 2.5),
                   lopo_hi=np.percentile(bs, 97.5), lopo_p=wilcoxon(lopo.matched, lopo.full).pvalue)
    rows.append(rec)
out = pd.DataFrame(rows)
out.to_csv(f"{R}/encoder_summary.csv", index=False)
print(out.round(3).to_string(index=False))
