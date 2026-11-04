"""Every encoder's full-data and matched structure embeddings against raw ECFP4+RF on the
identical membership-control splits (positives = class, negatives from M), Phase 2 and 1.
RF: 200 trees, max depth 6 (as in the main baselines). Writes results/encoders_vs_ecfp4rf.csv."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from run_experiments import DATA, RESULTS, CLASS_COLS
import membership_control_v2 as mc

fp = np.load(f"{DATA}/drug_ecfp4.npy").astype(np.float32)
rows = []
for ds in ["p2", "p1"]:
    for ci, cls in enumerate(CLASS_COLS):
        short = cls.replace("class_", "")
        z = np.load(f"{RESULTS}/membership_v2_embeddings/ecfp4_{ds}_{short}.npz")
        sp = mc.probe_splits(z["pos"], z["M"])
        rf = []
        for a, b, c, d in sp:
            m = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0, n_jobs=8,
                                       class_weight="balanced").fit(fp[a], b)
            rf.append(roc_auc_score(d, m.predict_proba(fp[c])[:, 1]))
        rf = np.array(rf)
        for inp in (["ecfp4", "chemberta", "gcn", "finlayson"] if ds == "p2" else ["ecfp4"]):
            per = pd.read_csv(f"{RESULTS}/membership_v2_{inp}_{ds}.csv")
            per = per[per.cls == short].sort_values("split")
            for arm in ["full", "matched"]:
                x = per[arm].values
                rows.append(dict(dataset=ds, cls=short, input=inp, arm=arm, auroc=x.mean(),
                                 ecfp4_rf=rf.mean(), diff=x.mean() - rf.mean(),
                                 p=wilcoxon(x, rf).pvalue))
out = pd.DataFrame(rows)
out.to_csv(f"{RESULTS}/encoders_vs_ecfp4rf.csv", index=False)
print(out.pivot_table(index=["dataset", "cls"], columns=["input", "arm"], values="diff").round(3).to_string())
print(out.pivot_table(index=["dataset", "cls"], columns=["input", "arm"], values="p").round(3).to_string())
print(out.groupby(["dataset", "cls"]).ecfp4_rf.first().round(3))
