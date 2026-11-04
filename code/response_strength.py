"""Per-drug transcriptional response strength in Phase 2 (for the AED hypothesis, H5).
norm: mean L2 norm of the drug's 978-gene level-5 signatures.
repcorr: mean pairwise Pearson correlation among up to 60 of the drug's signatures
(replicate consistency; near 0 means the drug's signatures look like noise).
Writes results/response_strength.csv."""
import numpy as np, pandas as pd
from run_experiments import DATA, RESULTS, CLASS_COLS
meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
expr = np.load(f"{DATA}/expr_signatures.npy", mmap_mode="r")
rng = np.random.RandomState(0)
rows = []
for d, idx in meta.groupby("drug_uid").indices.items():
    idx = np.sort(rng.choice(idx, min(60, len(idx)), replace=False))
    X = np.asarray(expr[idx], dtype=np.float64)
    norm = np.linalg.norm(X, axis=1).mean()
    if len(idx) > 1:
        C = np.corrcoef(X); rc = C[np.triu_indices(len(idx), 1)].mean()
    else:
        rc = np.nan
    r = dict(drug_uid=d, n_sigs=len(meta.groupby("drug_uid").indices[d]) if False else len(idx), norm=norm, repcorr=rc)
    rows.append(r)
out = pd.DataFrame(rows)
lab = meta.drop_duplicates("drug_uid").set_index("drug_uid")[["pert_iname"] + CLASS_COLS]
out = out.join(lab, on="drug_uid")
out.to_csv(f"{RESULTS}/response_strength.csv", index=False)
bg = out[out[CLASS_COLS].sum(1) == 0]
print("background median norm %.2f repcorr %.3f" % (bg.norm.median(), bg.repcorr.median()))
for c in CLASS_COLS:
    s = out[out[c] == 1]
    print(f"{c:22s} n={len(s):2d} norm={s.norm.median():.2f} repcorr={s.repcorr.median():.3f} "
          f"pct_rank_repcorr={(bg.repcorr < s.repcorr.median()).mean():.2f}")
