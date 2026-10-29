"""Reviewer W3: Table 3's random-holdout control used a single set of 3
pretraining runs, each removing avg_class_size=8 random drugs (the mean class
size across all six classes), reused as the comparison point for every class.
For AED (16 positives) the control therefore removed half as many drugs as
the real class; for statin (5 positives) it removed more drugs than the class
has members. Only classes near the average (NSAID=8, beta-blocker=6,
antipsychotic=7) were reasonably matched. The draw pool was also unrestricted,
so a random holdout could by chance include a few of the target class's own
positives.

This retrains a dedicated random-holdout control per class: 3 independent
seeds, holdout size exactly equal to that class's own positive count, drawn
only from compounds NOT in that class (so the control cannot accidentally
remove real class members), then reruns the full downstream comparison
(full vs. per-class random-holdout vs. class-holdout, concatenated and
structure-only probes) on the same repeated drug-disjoint splits used
elsewhere.

Writes results/random_holdout_per_class.csv.
"""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, ".")
from run_experiments import pretrain, embed_drugs
from splits import repeated_subsample_splits

DATA = "../data"
RESULTS = "../results"
LOGF = "../logs/random_holdout_per_class.log"
CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]
N_SEEDS = 3


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


def eval_probe(train_X, train_y, test_X, test_y):
    if len(np.unique(train_y)) < 2:
        return None
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(train_X, train_y)
    score = clf.predict_proba(test_X)[:, 1]
    try:
        return roc_auc_score(test_y, score)
    except ValueError:
        return None


def main():
    t0 = time.time()
    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    expr = np.load(f"{DATA}/expr_signatures.npy")
    ecfp = np.load(f"{DATA}/drug_ecfp4.npy").astype(np.float32)
    drug_uid_of_sig = meta["drug_uid"].values.astype(int)
    sig_indices_all = np.arange(len(meta))
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_drug_uids = np.array(sorted(drug_level.index.tolist()))

    rows = []
    for cls in CLASS_COLS:
        short = cls.replace("class_", "")
        pos_series = drug_level[cls]
        pos_drugs_set = set(pos_series[pos_series == 1].index.tolist())
        n_pos = len(pos_drugs_set)
        bg_pool = np.array([d for d in all_drug_uids if d not in pos_drugs_set])

        pos_drugs = np.array(sorted(pos_drugs_set))
        pos_drugs = pos_drugs[np.isin(pos_drugs, all_drug_uids)]
        splits = repeated_subsample_splits(pos_drugs, all_drug_uids, n_repeats=10, seed0=42)

        embeds = []
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(6000 + seed)
            holdout = set(rng.choice(bg_pool, size=n_pos, replace=False).tolist())
            keep_mask = ~np.isin(drug_uid_of_sig, list(holdout))
            sig_idx_keep = sig_indices_all[keep_mask]
            m, _ = pretrain(sig_idx_keep, drug_uid_of_sig, ecfp, expr,
                             seed=6000 + seed, log_tag=f"randholdout_perclass:{short}:{seed}")
            z_s, z_e, has_e = embed_drugs(m, all_drug_uids, ecfp, expr, drug_uid_of_sig, sig_indices_all)
            assert has_e.all(), "BUG: zero-vector embedding artifact"
            embeds.append({"z_s": z_s, "z_e": z_e})
        log(f"{short}: n_pos={n_pos}, trained {N_SEEDS} size-matched random-holdout models "
            f"({time.time()-t0:.0f}s elapsed)")

        for split in splits:
            seen, test, test_y = split["seen_drugs"], split["test_drugs"], split["test_labels"]
            seen_y = pos_series.reindex(seen).fillna(0).values
            for seed, e in enumerate(embeds):
                z_concat = np.concatenate([e["z_s"], e["z_e"]], axis=1)
                a_concat = eval_probe(z_concat[seen], seen_y, z_concat[test], test_y)
                a_struct = eval_probe(e["z_s"][seen], seen_y, e["z_s"][test], test_y)
                if a_concat is not None:
                    rows.append({"class": short, "seed": split["seed"], "pretrain_seed": seed,
                                 "view": "concat", "auroc": a_concat})
                if a_struct is not None:
                    rows.append({"class": short, "seed": split["seed"], "pretrain_seed": seed,
                                 "view": "structure", "auroc": a_struct})

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/random_holdout_per_class.csv", index=False)
    summary = out.groupby(["class", "view"]).auroc.mean()
    log(str(summary))
    log(f"\nWrote {RESULTS}/random_holdout_per_class.csv, DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
