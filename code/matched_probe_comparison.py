"""Reviewer W4: the ChemBERTa baseline was previously evaluated only with a
random forest, while the contrastive encoder was only evaluated with a
logistic-regression probe, so the comparison in Table 1 confounded
representation with downstream classifier. This applies BOTH probes (logistic
regression and random forest) to all three representations (raw ECFP4, frozen
ChemBERTa, frozen contrastive-full structure embedding), under the identical
repeated drug-disjoint splits, isolating the representation's contribution.

No retraining: ECFP4 is already on disk, ChemBERTa embeddings were computed in
chemberta_baseline.py, and the contrastive structure embedding is the already-
saved results/z_s_full.npy (full-regime, pseed=0), already used for Table 1's
"Contrastive" column (there as concat with expression; here structure-only, to
match ChemBERTa and ECFP4, which are structure-only representations).

Writes results/matched_probe_comparison.csv.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, ".")
from splits import repeated_subsample_splits

DATA = "../data"
RESULTS = "../results"
CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]


def fit_eval(train_X, train_y, test_X, test_y, probe):
    if len(np.unique(train_y)) < 2:
        return None
    if probe == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    else:
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=0)
    clf.fit(train_X, train_y)
    score = clf.predict_proba(test_X)[:, 1]
    try:
        return roc_auc_score(test_y, score)
    except ValueError:
        return None


def main():
    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_drug_uids = np.array(sorted(drug_level.index.tolist()))

    ecfp = np.load(f"{DATA}/drug_ecfp4.npy").astype(np.float32)
    chemberta = np.load(f"{RESULTS}/chemberta_embeddings.npy")
    contrastive = np.load(f"{RESULTS}/z_s_full.npy")
    reps = {"ECFP4": ecfp, "ChemBERTa": chemberta, "Contrastive-structure": contrastive}

    rows = []
    for cls in CLASS_COLS:
        pos_series = drug_level[cls]
        pos_drugs = np.array(sorted(pos_series[pos_series == 1].index.tolist()))
        pos_drugs = pos_drugs[np.isin(pos_drugs, all_drug_uids)]
        if len(pos_drugs) < 3:
            continue
        splits = repeated_subsample_splits(pos_drugs, all_drug_uids, n_repeats=10, seed0=42)
        for split in splits:
            seen, test, test_y = split["seen_drugs"], split["test_drugs"], split["test_labels"]
            seen_y = pos_series.reindex(seen).fillna(0).values
            for rep_name, X in reps.items():
                for probe in ("logreg", "rf"):
                    a = fit_eval(X[seen], seen_y, X[test], test_y, probe)
                    if a is not None:
                        rows.append({"class": cls, "seed": split["seed"], "representation": rep_name,
                                     "probe": probe, "auroc": a})
        print(f"{cls}: done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/matched_probe_comparison.csv", index=False)
    summary = out.groupby(["representation", "probe"]).auroc.agg(["mean", "std"])
    print(summary)
    print(f"\nWrote {RESULTS}/matched_probe_comparison.csv")


if __name__ == "__main__":
    main()
