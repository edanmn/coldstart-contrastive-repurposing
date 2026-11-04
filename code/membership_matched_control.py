"""Membership-matched control for the class-holdout effect.

In the standard class-holdout protocol every probe positive (the held-out
class) is absent from pretraining, while every probe negative was present.
A probe can then score well by detecting "not seen by the encoder" rather
than class membership. This script removes that difference.

For each class C we draw a fixed set M of 400 non-class drugs and compare,
under one probe protocol whose negatives all come from M:
  full     : nothing removed            (positives seen,   negatives seen)
  holdout  : C removed                  (positives unseen, negatives seen)
  matched  : C and M removed            (positives unseen, negatives unseen)
If the holdout gain over full comes from membership, matched minus full
should shrink toward zero (and to chance for random input vectors).

Also reports, for the matched encoder, a pure membership probe: unseen
non-class drugs (from M) vs seen non-class drugs, no class labels involved.

Reuses pretrain()/embed_drugs()/eval_probe() from run_experiments.py.
Usage: python3 membership_matched_control.py {ecfp4|random_control}
Writes results/membership_matched_<feat>.csv (per split) and _summary.csv.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from run_experiments import pretrain, embed_drugs, eval_probe, CLASS_COLS, DATA, RESULTS
from fixed_featurization_replication import build_random

N_PRETRAIN_SEEDS = 3
N_REPEATS = 10
M_SIZE = 400
TEST_NEG = 150
TEST_POS_FRAC = 0.35


def probe_splits(pos, M, seed0=42):
    out = []
    for r in range(N_REPEATS):
        rng = np.random.RandomState(seed0 + r)
        n_tp = max(2, int(round(len(pos) * TEST_POS_FRAC)))
        n_tp = min(n_tp, len(pos) - 1)
        tp = rng.choice(pos, n_tp, replace=False)
        tn = rng.choice(M, TEST_NEG, replace=False)
        trp = np.setdiff1d(pos, tp)
        trn = np.setdiff1d(M, tn)
        out.append((np.concatenate([trp, trn]),
                    np.concatenate([np.ones(len(trp)), np.zeros(len(trn))]),
                    np.concatenate([tp, tn]),
                    np.concatenate([np.ones(len(tp)), np.zeros(len(tn))])))
    return out


def membership_splits(M, seen_pool, seed0=42):
    """Positives: unseen non-class drugs (M). Negatives: seen non-class drugs."""
    out = []
    for r in range(N_REPEATS):
        rng = np.random.RandomState(seed0 + r)
        tp = rng.choice(M, TEST_NEG, replace=False)
        tn = rng.choice(seen_pool, TEST_NEG, replace=False)
        trp = np.setdiff1d(M, tp)
        trn = rng.choice(np.setdiff1d(seen_pool, tn), len(trp), replace=False)
        out.append((np.concatenate([trp, trn]),
                    np.concatenate([np.ones(len(trp)), np.zeros(len(trn))]),
                    np.concatenate([tp, tn]),
                    np.concatenate([np.ones(len(tp)), np.zeros(len(tn))])))
    return out


def score(Zs, splits):
    """Mean AUROC over pretraining seeds, per split."""
    per = []
    for trX, trY, teX, teY in splits:
        v = [eval_probe(Z[trX], trY, Z[teX], teY)["auroc"] for Z in Zs]
        per.append(np.mean(v))
    return np.array(per)


def main():
    feat_name = sys.argv[1]
    t0 = time.time()
    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    expr = np.load(f"{DATA}/expr_signatures.npy")
    ecfp = np.load(f"{DATA}/drug_ecfp4.npy").astype(np.float32)
    feat = ecfp if feat_name == "ecfp4" else build_random(ecfp.shape[0])
    drug_uid_of_sig = meta["drug_uid"].values.astype(int)
    sig_all = np.arange(len(meta))
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_drugs = np.array(sorted(drug_level.index.tolist()))

    def train_embed(exclude, seed, tag):
        keep = sig_all[~np.isin(drug_uid_of_sig, list(exclude))]
        m, _ = pretrain(keep, drug_uid_of_sig, feat, expr, seed=seed, log_tag=tag)
        z_s, _, has_e = embed_drugs(m, all_drugs, feat, expr, drug_uid_of_sig, sig_all)
        assert has_e.all()
        return z_s

    full = [train_embed(set(), s, f"{feat_name}:full:{s}") for s in range(N_PRETRAIN_SEEDS)]
    rows, summ = [], []
    for ci, cls in enumerate(CLASS_COLS):
        lab = drug_level[cls].reindex(all_drugs).fillna(0).values
        pos = all_drugs[lab == 1]
        nonclass = all_drugs[lab == 0]
        M = np.sort(np.random.RandomState(900 + ci).choice(nonclass, M_SIZE, replace=False))
        seen_nonclass = np.setdiff1d(nonclass, M)
        hold = [train_embed(set(pos), 2000 + s, f"{feat_name}:hold:{cls}:{s}")
                for s in range(N_PRETRAIN_SEEDS)]
        matched = [train_embed(set(pos) | set(M), 3000 + s, f"{feat_name}:matched:{cls}:{s}")
                   for s in range(N_PRETRAIN_SEEDS)]
        sp = probe_splits(pos, M)
        f_, h_, m_ = score(full, sp), score(hold, sp), score(matched, sp)
        mem = score(matched, membership_splits(M, seen_nonclass))
        mem_full = score(full, membership_splits(M, seen_nonclass))
        short = cls.replace("class_", "")
        for r in range(N_REPEATS):
            rows.append(dict(featurization=feat_name, cls=short, split=r, full=f_[r],
                             holdout=h_[r], matched=m_[r], membership_matched=mem[r],
                             membership_full=mem_full[r]))
        def p(a, b):
            return float("nan") if np.allclose(a, b) else wilcoxon(a, b).pvalue
        summ.append(dict(featurization=feat_name, cls=short, full=f_.mean(), holdout=h_.mean(),
                         matched=m_.mean(), holdout_minus_full=h_.mean() - f_.mean(),
                         matched_minus_full=m_.mean() - f_.mean(),
                         p_holdout_vs_full=p(h_, f_), p_matched_vs_full=p(m_, f_),
                         p_matched_vs_holdout=p(m_, h_),
                         membership_auroc_matched=mem.mean(), membership_auroc_full=mem_full.mean()))
        print(f"[{feat_name}] {short}: full={f_.mean():.3f} hold={h_.mean():.3f} "
              f"matched={m_.mean():.3f} membership={mem.mean():.3f} "
              f"(full enc {mem_full.mean():.3f}) {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{RESULTS}/membership_matched_{feat_name}.csv", index=False)
        pd.DataFrame(summ).to_csv(f"{RESULTS}/membership_matched_{feat_name}_summary.csv", index=False)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
