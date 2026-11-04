"""H5b: does the contrastive encoder keep fingerprint similarity for compounds it never
saw, and lose it for compounds it trained on?
For each class encoder in membership_v2 (input ecfp4 or random, dataset p2):
  rho_unseen : Spearman(embedding cosine, ECFP4 Tanimoto) over pairs within M,
               on the matched encoder (M unseen)
  rho_seen   : same statistic, same pairs, on the full encoder (M seen)
  rho_class_full / rho_class_matched : same, over pairs within the class
Writes results/similarity_preservation_<input>.csv."""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from run_experiments import DATA, RESULTS, CLASS_COLS

inp = sys.argv[1] if len(sys.argv) > 1 else "ecfp4"
fp = np.load(f"{DATA}/drug_ecfp4.npy").astype(np.float32)
def tanimoto(ix):
    X = fp[ix]; inter = X @ X.T; cnt = X.sum(1)
    return inter / (cnt[:, None] + cnt[None, :] - inter + 1e-9)
def cos(Z):
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True); return Z @ Z.T
def rho(Z, ix):
    iu = np.triu_indices(len(ix), 1)
    return spearmanr(cos(Z[ix].astype(np.float32))[iu], tanimoto(ix)[iu]).correlation
E = f"{RESULTS}/membership_v2_embeddings/{inp}_p2"
full = np.load(f"{E}_full.npy")
rows = []
for cls in CLASS_COLS:
    short = cls.replace("class_", "")
    try:
        z = np.load(f"{E}_{short}.npz")
    except FileNotFoundError:
        continue
    M, pos = z["M"], z["pos"]
    for s in range(full.shape[0]):
        rows.append(dict(input=inp, cls=short, seed=s,
                         rho_M_seen_full=rho(full[s], M), rho_M_unseen_matched=rho(z["matched"][s], M),
                         rho_M_seen_hold=rho(z["hold"][s], M),
                         rho_class_full=rho(full[s], pos), rho_class_matched=rho(z["matched"][s], pos),
                         rho_class_hold=rho(z["hold"][s], pos)))
d = pd.DataFrame(rows)
d.to_csv(f"{RESULTS}/similarity_preservation_{inp}.csv", index=False)
g = d.groupby("cls").mean(numeric_only=True).drop(columns="seed")
print(g.round(3).to_string())
if len(g) >= 5:
    print("unseen vs seen (M pairs), Wilcoxon over class x seed:",
          wilcoxon(d.rho_M_unseen_matched, d.rho_M_seen_hold).pvalue)
