"""Reviewer W3: the paper's downstream task is class-membership recovery,
which the manuscript now states clearly is not the same as a drug-disease
efficacy prediction. This adds an external, independent biological check:
SIDER adverse-event profiles, which were never used anywhere in pretraining,
the class labels, or the structure/expression features, and which are a real,
clinically reported pharmacological signal (two drugs sharing more adverse
events than chance is evidence of shared pharmacology, independent of
structure).

For each class, restricted to the 534 LINCS compounds with SIDER coverage
(534/534 side-effect pairs resolve; coverage is 47/48 class positives, data/
lincs_sider_map.csv + data/sider_pairs.csv), we ask: among a positive drug's
5 nearest neighbors in STRUCTURE EMBEDDING space (full-data vs class-holdout
pretraining), how many adverse events does the query drug share with its
neighbors on average (Jaccard over SIDER term sets), compared to a random-5
background? This tests whether the representation that the class-holdout
regime produces picks out genuinely pharmacologically related compounds by an
independent measure, not just class-label-matched ones.

Requires results/z_s_full.npy (already on disk) and results/
z_s_classholdout_<class>.npy (written by drug_level_and_embedding_analysis.py).

Writes results/sider_biological_validation.csv.
"""
import numpy as np
import pandas as pd

DATA = "../data"
RESULTS = "../results"
CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]
K = 5
N_BG_DRAWS = 200
RNG_SEED = 909


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def main():
    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_drug_uids = np.array(sorted(drug_level.index.tolist()))

    lm = pd.read_csv(f"{DATA}/lincs_sider_map.csv")
    sp = pd.read_csv(f"{DATA}/sider_pairs.csv")
    cid_to_se = sp.groupby("cid_flat")["se_name"].apply(set).to_dict()
    drug_se = {}
    for _, row in lm.iterrows():
        se = cid_to_se.get(row["cid"])
        if se:
            drug_se[int(row["drug_uid"])] = se
    sider_drugs = np.array(sorted(drug_se.keys()))
    sider_drugs = sider_drugs[np.isin(sider_drugs, all_drug_uids)]
    print(f"SIDER-covered drugs usable: {len(sider_drugs)}", flush=True)

    z_s_full = np.load(f"{RESULTS}/z_s_full.npy")
    rng = np.random.RandomState(RNG_SEED)

    rows = []
    for cls in CLASS_COLS:
        short = cls.replace("class_", "")
        try:
            z_s_cold = np.load(f"{RESULTS}/z_s_classholdout_{short}.npy")
        except FileNotFoundError:
            print(f"  {short}: no class-holdout embedding on disk yet, skipping", flush=True)
            continue

        pos_series = drug_level[cls]
        pos_drugs = np.array(sorted(pos_series[pos_series == 1].index.tolist()))
        pos_sider = pos_drugs[np.isin(pos_drugs, sider_drugs)]
        if len(pos_sider) < 3:
            continue

        sider_idx_set = set(sider_drugs.tolist())

        def nn_jaccard(z, query):
            sims = z[sider_drugs] @ z[query]
            order = np.argsort(-sims)
            neighbors = [sider_drugs[i] for i in order if sider_drugs[i] != query][:K]
            return np.mean([jaccard(drug_se[query], drug_se[n]) for n in neighbors])

        def bg_jaccard(query):
            draws = []
            pool = [d for d in sider_drugs if d != query]
            for _ in range(N_BG_DRAWS):
                sample = rng.choice(pool, size=K, replace=False)
                draws.append(np.mean([jaccard(drug_se[query], drug_se[s]) for s in sample]))
            return np.mean(draws)

        full_vals, cold_vals, bg_vals = [], [], []
        for d in pos_sider:
            full_vals.append(nn_jaccard(z_s_full, d))
            cold_vals.append(nn_jaccard(z_s_cold, d))
            bg_vals.append(bg_jaccard(d))

        rows.append(dict(cls=short, n_pos_sider=len(pos_sider),
                          nn5_jaccard_full=np.mean(full_vals),
                          nn5_jaccard_classholdout=np.mean(cold_vals),
                          nn5_jaccard_random_background=np.mean(bg_vals)))
        print(f"{short}: n={len(pos_sider)} full={np.mean(full_vals):.4f} "
              f"cold={np.mean(cold_vals):.4f} bg={np.mean(bg_vals):.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/sider_biological_validation.csv", index=False)
    print(f"\nWrote {RESULTS}/sider_biological_validation.csv")


if __name__ == "__main__":
    main()
