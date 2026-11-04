"""Per-pretraining-seed stability of the membership signal and the drug-level gain.
For each input with 3 seeds in membership_v2_embeddings (dataset p2):
  mem_auroc[seed] : membership-probe AUROC on the matched encoder of that seed
                    (same M / seen-pool splits as membership_control_v2.py), mean over classes
  lopo_delta[seed]: drug-level matched-minus-full percentile rank using only that seed's
                    encoders (same LOPO protocol as lopo_matched.py), pooled over 48 drugs,
                    with a 5,000-resample bootstrap CI over drugs
Usage: python3 seed_stability.py <input> [<input> ...]. Writes results/seed_stability.csv
(rows appended/replaced per input)."""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from run_experiments import RESULTS, CLASS_COLS
from membership_control_v2 import load_dataset, membership_splits, score

expr, uid_of_sig, drug_level = load_dataset("p2")
universe = np.array(sorted(set(uid_of_sig.tolist())))
OUT = f"{RESULTS}/seed_stability.csv"
rows = []
for inp in sys.argv[1:]:
    E = f"{RESULTS}/membership_v2_embeddings/{inp}_p2"
    full = np.load(f"{E}_full.npy").astype(np.float32)
    ns = full.shape[0]
    mem = np.zeros((ns, len(CLASS_COLS)))
    dm = [[] for _ in range(ns)]
    for ci, cls in enumerate(CLASS_COLS):
        short = cls.replace("class_", "")
        z = np.load(f"{E}_{short}.npz")
        pos, M, M2 = z["pos"], z["M"], z["M2"]
        matched = z["matched"].astype(np.float32)
        lab = drug_level[cls].reindex(universe).fillna(0).values
        nonclass = universe[lab == 0]
        seen = np.setdiff1d(np.setdiff1d(nonclass, M), M2)
        msp = membership_splits(M, seen)
        rng = np.random.RandomState(500 + ci)
        Mp = rng.permutation(M); half = len(M) // 2
        neg_tr, bg = Mp[:half], Mp[half:]
        for s in range(ns):
            mem[s, ci] = score([matched[s]], msp).mean()
            for d in pos:
                others = pos[pos != d]
                rk = {}
                for arm, Z in [("full", full[s]), ("matched", matched[s])]:
                    X = np.vstack([Z[others], Z[neg_tr]])
                    y = np.r_[np.ones(len(others)), np.zeros(len(neg_tr))]
                    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
                    rk[arm] = (clf.decision_function(Z[bg]) < clf.decision_function(Z[[d]])[0]).mean()
                dm[s].append(rk["matched"] - rk["full"])
    for s in range(ns):
        v = np.array(dm[s])
        bs = [np.random.RandomState(i).choice(v, len(v)).mean() for i in range(5000)]
        rows.append(dict(input=inp, seed=s, mem_min=mem[s].min(), mem_max=mem[s].max(),
                         mem_mean=mem[s].mean(), lopo_n=len(v), lopo_delta=v.mean(),
                         lopo_lo=np.percentile(bs, 2.5), lopo_hi=np.percentile(bs, 97.5)))
        print(rows[-1], flush=True)
new = pd.DataFrame(rows)
if os.path.exists(OUT):
    old = pd.read_csv(OUT)
    new = pd.concat([old[~old.input.isin(new.input)], new])
new.to_csv(OUT, index=False)
print(new.round(3).to_string(index=False))
