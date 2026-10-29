"""Reviewer-requested direct measurement of same-class false-negative collision
frequency (W3) and within-class chemical diversity, correlated with the observed
holdout improvement (W5). Both computed analytically/combinatorially from
data/drug_ecfp4.npy and data/dataset_meta.parquet: no retraining.

W3: for a signature of a class-c drug, the expected fraction of a random
InfoNCE batch (B=256, drawn without replacement from all 103,989 signatures)
that consists of OTHER class-c drugs' signatures (same-class, different-drug:
exactly the pairs masked out under "exclude" and added to positives under
"positive" in models.py). This is an exact hypergeometric expectation, not a
simulation, and needs no trained model.

W5: mean pairwise Tanimoto similarity among class-c drugs' ECFP4 fingerprints
(within-class), versus mean Tanimoto similarity from a class-c drug to a random
background drug (class-to-background). Spearman-correlates both this and the
collision rate against the structure-view cold-start delta (results/coldstart_significance.csv)
and the mechanism-test recovery percentage (results/mechanism_test_structure.csv).

Writes results/chemistry_and_collision.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

D = "../data/"
R = "../results/"
CLASSES = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
           "class_BetaBlocker", "class_Antipsychotic"]
BATCH_SIZE = 256


def tanimoto_matrix(fp):
    fp = fp.astype(bool)
    inter = fp.astype(np.int32) @ fp.astype(np.int32).T
    row_sums = fp.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return sim


def main():
    ecfp = np.load(D + "drug_ecfp4.npy").astype(np.float32)
    meta = pd.read_parquet(D + "dataset_meta.parquet")
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    n_drugs = ecfp.shape[0]
    sim = tanimoto_matrix(ecfp)
    sig_count_by_drug = meta.groupby("drug_uid").size()
    N_total_sigs = len(meta)

    rows = []
    for cls in CLASSES:
        pos_drugs = drug_level.index[drug_level[cls] == 1].tolist()
        pos_drugs = [d for d in pos_drugs if d < n_drugs]
        bg_drugs = [d for d in range(n_drugs) if d not in pos_drugs]

        # W5: within-class vs class-to-background Tanimoto similarity.
        if len(pos_drugs) >= 2:
            sub = sim[np.ix_(pos_drugs, pos_drugs)]
            iu = np.triu_indices(len(pos_drugs), k=1)
            within_sim = sub[iu].mean()
        else:
            within_sim = float("nan")
        cross = sim[np.ix_(pos_drugs, bg_drugs)]
        to_bg_sim = cross.mean()

        # W3: exact expected same-class-different-drug collision rate per
        # anchor signature, under a uniform random batch of size B drawn
        # without replacement from all N signatures (hypergeometric mean).
        class_sig_counts = sig_count_by_drug.reindex(pos_drugs).fillna(0).astype(int)
        total_class_sigs = int(class_sig_counts.sum())
        # Weight by each drug's own signature count (anchors are signatures,
        # not drugs): for an anchor signature of drug d, the number of
        # "other same-class" signatures available is total_class_sigs - count(d).
        other_class_sigs = total_class_sigs - class_sig_counts
        # weighted mean over anchors (each drug contributes count(d) anchors)
        expected_other_per_anchor = (other_class_sigs * class_sig_counts).sum() / max(total_class_sigs, 1)
        # Expected COUNT (not a probability) of same-class, different-drug
        # collision partners landing in a random anchor's batch, hypergeometric mean.
        expected_collisions_per_anchor = (BATCH_SIZE - 1) * expected_other_per_anchor / (N_total_sigs - 1)

        rows.append(dict(
            cls=cls.replace("class_", ""), n_pos=len(pos_drugs),
            total_class_sigs=total_class_sigs,
            within_class_tanimoto=within_sim, class_to_bg_tanimoto=to_bg_sim,
            diversity_ratio=within_sim / to_bg_sim if to_bg_sim > 0 else float("nan"),
            expected_collisions_per_anchor_per_batch=expected_collisions_per_anchor,
        ))

    out = pd.DataFrame(rows)

    # Join with cold-start structure-view delta and mechanism recovery%.
    cs = pd.read_csv(R + "coldstart_significance.csv")
    cs["cls"] = cs["class"].str.replace("class_", "", regex=False)
    struct = cs[cs.view == "structure"].set_index("cls")["classholdout_minus_full"]
    out["structure_delta"] = out["cls"].map(struct)

    mech = pd.read_csv(R + "mechanism_test_structure.csv")
    mech["cls"] = mech["cls"].str.replace("class_", "", regex=False)
    mech = mech.set_index("cls")["pct_recovery"]
    out["mechanism_recovery_pct"] = out["cls"].map(mech)

    out.to_csv(R + "chemistry_and_collision.csv", index=False)
    pd.set_option("display.width", 160)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Spearman correlations across the 6 classes (n=6, weak power, report as exploratory) ===")
    for feat in ["within_class_tanimoto", "diversity_ratio", "expected_collisions_per_anchor_per_batch"]:
        r1, p1 = spearmanr(out[feat], out["structure_delta"])
        r2, p2 = spearmanr(out[feat], out["mechanism_recovery_pct"])
        print(f"{feat:32s} vs structure_delta: rho={r1:+.3f} p={p1:.3f}   vs mechanism_recovery%: rho={r2:+.3f} p={p2:.3f}")


if __name__ == "__main__":
    main()
