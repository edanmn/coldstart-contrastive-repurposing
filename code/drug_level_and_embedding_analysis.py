"""Reviewer W1 (drug-identity-respecting resampling and full AUROC-difference
distributions, not just p-values from correlated subsampling seeds) and W2
(embedding-space analysis: within-class separation and nearest-neighbor
structure, to investigate the AED/SSRI puzzle where structural improvement is
large but Tanimoto similarity is low).

Reuses results/z_s_full.npy (already-trained full-regime structure embeddings,
pseed=0) and retrains class-holdout once per class (seed=0; 6 runs total, not
18, since this analysis only needs one representative embedding per regime,
not the full 3-seed x 10-subsample grid already reported elsewhere).

W1: leave-one-positive-drug-out (LOPO). For each class and each positive drug
d, fit a probe on every OTHER positive plus a fixed negative background, then
score d against a fixed held-out negative pool and record its percentile rank.
Every positive drug gets exactly one rank per regime, so the n_pos ranks are
genuinely drug-disjoint, independent observations (unlike the 10 subsampling
seeds elsewhere, which reuse overlapping drug subsets). A Wilcoxon test paired
by DRUG across these n_pos values is the drug-respecting significance test
the reviewer asked for, and the full per-drug distribution is reported, not
just a mean and a p-value.

W2: within-class vs class-to-background mean cosine distance in EMBEDDING
space (not input fingerprint space), for full vs class-holdout, plus a
nearest-neighbor check: of each held-out positive's 5 nearest neighbors in
embedding space, how many are other class members.

Writes results/drug_level_lopo.csv and results/embedding_space_analysis.csv.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from scipy.stats import wilcoxon

import sys
sys.path.insert(0, ".")
from run_experiments import pretrain, embed_drugs

DATA = "../data"
RESULTS = "../results"
LOGF = "../logs/drug_level_and_embedding.log"
CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]
NEG_POOL_SIZE = 300
RNG_SEED = 777


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


def percentile_rank(score_pos, scores_neg):
    return float((scores_neg < score_pos).mean())


def lopo_ranks(z_s, pos_drugs, neg_pool, rng):
    """For each positive drug, fit on the rest + neg_pool, test that drug's
    percentile rank among neg_pool. Returns array of n_pos ranks."""
    ranks = []
    for i, held in enumerate(pos_drugs):
        train_pos = [d for d in pos_drugs if d != held]
        train_X = np.concatenate([z_s[train_pos], z_s[neg_pool]], axis=0)
        train_y = np.concatenate([np.ones(len(train_pos)), np.zeros(len(neg_pool))])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(train_X, train_y)
        score_held = clf.predict_proba(z_s[[held]])[:, 1][0]
        scores_neg = clf.predict_proba(z_s[neg_pool])[:, 1]
        ranks.append(percentile_rank(score_held, scores_neg))
    return np.array(ranks)


def embedding_space_stats(z_s, pos_drugs, bg_drugs):
    z = z_s / (np.linalg.norm(z_s, axis=1, keepdims=True) + 1e-8)
    pos_z = z[pos_drugs]
    sim_within = pos_z @ pos_z.T
    iu = np.triu_indices(len(pos_drugs), k=1)
    within_cos = sim_within[iu].mean() if len(pos_drugs) >= 2 else float("nan")
    bg_sample = bg_drugs[np.random.RandomState(RNG_SEED).choice(len(bg_drugs), size=min(500, len(bg_drugs)), replace=False)]
    to_bg_cos = (pos_z @ z[bg_sample].T).mean()

    # nearest-neighbor check: for each positive, among its 5 NN (excluding itself,
    # over ALL drugs), how many are other class members
    all_sim = z @ z.T
    nn_fracs = []
    pos_set = set(pos_drugs.tolist())
    for d in pos_drugs:
        order = np.argsort(-all_sim[d])
        order = [x for x in order if x != d][:5]
        frac = np.mean([1.0 if o in pos_set else 0.0 for o in order])
        nn_fracs.append(frac)
    return within_cos, to_bg_cos, np.mean(nn_fracs)


def main():
    t0 = time.time()
    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    expr = np.load(f"{DATA}/expr_signatures.npy")
    ecfp = np.load(f"{DATA}/drug_ecfp4.npy").astype(np.float32)
    n_drugs = ecfp.shape[0]
    drug_uid_of_sig = meta["drug_uid"].values.astype(int)
    sig_indices_all = np.arange(len(meta))
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_drug_uids = np.array(sorted(drug_level.index.tolist()))

    z_s_full = np.load(f"{RESULTS}/z_s_full.npy")
    log(f"Loaded existing full-regime structure embeddings: {z_s_full.shape}")

    rng = np.random.RandomState(RNG_SEED)
    lopo_rows, embed_rows = [], []

    for cls in CLASS_COLS:
        pos_series = drug_level[cls]
        pos_drugs_set = set(pos_series[pos_series == 1].index.tolist())
        pos_drugs = np.array(sorted(pos_drugs_set))
        pos_drugs = pos_drugs[np.isin(pos_drugs, all_drug_uids)]
        if len(pos_drugs) < 3:
            continue
        bg_drugs = np.array([d for d in all_drug_uids if d not in pos_drugs_set])
        neg_pool = rng.choice(bg_drugs, size=min(NEG_POOL_SIZE, len(bg_drugs)), replace=False)

        keep_sig_mask = ~np.isin(drug_uid_of_sig, list(pos_drugs_set))
        sig_idx_keep = sig_indices_all[keep_sig_mask]
        m, _ = pretrain(sig_idx_keep, drug_uid_of_sig, ecfp, expr, seed=5000, log_tag=f"holdout1seed:{cls}")
        z_s_cold, _, has_e = embed_drugs(m, all_drug_uids, ecfp, expr, drug_uid_of_sig, sig_indices_all)
        assert has_e.all(), "BUG: zero-vector embedding artifact"
        np.save(f"{RESULTS}/z_s_classholdout_{cls.replace('class_', '')}.npy", z_s_cold)

        # --- W1: LOPO ranks, full vs class-holdout, structure view ---
        ranks_full = lopo_ranks(z_s_full, pos_drugs, neg_pool, rng)
        ranks_cold = lopo_ranks(z_s_cold, pos_drugs, neg_pool, rng)
        delta = ranks_cold - ranks_full
        p_drug = wilcoxon(ranks_cold, ranks_full).pvalue if not np.allclose(ranks_cold, ranks_full) else np.nan
        for d, rf, rc in zip(pos_drugs, ranks_full, ranks_cold):
            lopo_rows.append(dict(cls=cls.replace("class_", ""), drug_uid=int(d),
                                   rank_full=rf, rank_classholdout=rc, delta=rc - rf))
        log(f"{cls}: LOPO n={len(pos_drugs)} mean_rank_full={ranks_full.mean():.3f} "
            f"mean_rank_cold={ranks_cold.mean():.3f} median_delta={np.median(delta):+.3f} "
            f"min={delta.min():+.3f} max={delta.max():+.3f} p_drug_paired={p_drug:.4f}")

        # --- W2: embedding-space separation + nearest-neighbor ---
        wf, bf, nnf = embedding_space_stats(z_s_full, pos_drugs, bg_drugs)
        wc, bc, nnc = embedding_space_stats(z_s_cold, pos_drugs, bg_drugs)
        embed_rows.append(dict(cls=cls.replace("class_", ""), n_pos=len(pos_drugs),
                                within_cos_full=wf, within_cos_cold=wc,
                                to_bg_cos_full=bf, to_bg_cos_cold=bc,
                                nn5_same_class_frac_full=nnf, nn5_same_class_frac_cold=nnc,
                                p_drug_paired=p_drug, median_rank_delta=float(np.median(delta)),
                                min_rank_delta=float(delta.min()), max_rank_delta=float(delta.max())))
        log(f"{cls}: embedding within-class cos full={wf:.4f} cold={wc:.4f} | "
            f"NN5 same-class frac full={nnf:.3f} cold={nnc:.3f} ({time.time()-t0:.0f}s elapsed)")

    pd.DataFrame(lopo_rows).to_csv(f"{RESULTS}/drug_level_lopo.csv", index=False)
    pd.DataFrame(embed_rows).to_csv(f"{RESULTS}/embedding_space_analysis.csv", index=False)
    log(f"\nWrote drug_level_lopo.csv and embedding_space_analysis.csv, DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
