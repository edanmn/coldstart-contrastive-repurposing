"""H4: drug-level, membership-free test. Leave-one-positive-out on the saved membership_v2
embeddings. Background = M, split once per class into 200 probe negatives and 200 test
background drugs. full: all seen; matched: positives and background all unseen; hold and
sizehold: positives unseen, background seen. Percentile rank of each held-out positive
among the test background, averaged over pretraining seeds.
Usage: python3 lopo_matched.py <input> <dataset>. Writes results/lopo_matched_<input>_<ds>.csv"""
import sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from run_experiments import RESULTS, CLASS_COLS

inp, ds = sys.argv[1], sys.argv[2]
E = f"{RESULTS}/membership_v2_embeddings/{inp}_{ds}"
full = np.load(f"{E}_full.npy").astype(np.float32)
rows = []
for ci, cls in enumerate(CLASS_COLS):
    short = cls.replace("class_", "")
    z = np.load(f"{E}_{short}.npz")
    pos, M = z["pos"], z["M"]
    rng = np.random.RandomState(500 + ci)
    Mp = rng.permutation(M); half = len(M) // 2
    neg_tr, bg = Mp[:half], Mp[half:]
    arms = {"full": full, **{k: z[k].astype(np.float32) for k in ["hold", "sizehold", "matched"]}}
    for d in pos:
        others = pos[pos != d]
        rec = dict(input=inp, dataset=ds, cls=short, drug_uid=int(d))
        for arm, Zs in arms.items():
            rk = []
            for Z in Zs:
                X = np.vstack([Z[others], Z[neg_tr]]); y = np.r_[np.ones(len(others)), np.zeros(len(neg_tr))]
                clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)
                s_d = clf.decision_function(Z[[d]])[0]; s_bg = clf.decision_function(Z[bg])
                rk.append((s_bg < s_d).mean())
            rec[arm] = float(np.mean(rk))
        rows.append(rec)
out = pd.DataFrame(rows)
out.to_csv(f"{RESULTS}/lopo_matched_{inp}_{ds}.csv", index=False)
def p(a, b): return wilcoxon(a, b).pvalue if not np.allclose(a, b) else np.nan
for c, g in out.groupby("cls", sort=False):
    print(f"{c:14s} n={len(g):2d} full={g.full.mean():.3f} matched={g.matched.mean():.3f} "
          f"hold={g.hold.mean():.3f} p(matched vs full)={p(g.matched, g.full):.4f}")
dm = out.matched - out.full
bs = [dm.sample(len(dm), replace=True, random_state=i).mean() for i in range(5000)]
print(f"POOLED n={len(out)} matched-full={dm.mean():+.3f} CI[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] "
      f"p={p(out.matched, out.full):.2e}; hold-full={(out.hold-out.full).mean():+.3f}")
for c in out.cls.unique():
    o = out[out.cls != c]; print(f"  drop {c}: matched-full={(o.matched-o.full).mean():+.3f} p={p(o.matched,o.full):.4f}")
