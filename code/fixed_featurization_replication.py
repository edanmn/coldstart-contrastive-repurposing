"""Strong-accept revision, highest-leverage fix: Section 4.7's structure-view
cold-start benefit (ECFP4+MLP) does not reproduce under any of four graph
(message-passing) encoder variants. Read in isolation that looks like "the
effect is an ECFP4 artifact." This script tests the sharper, more specific
hypothesis a RECOMB-style reviewer (and a literature check against the
fixed-vs-learned-representation generalization literature, e.g. the
fixed-fingerprint-vs-GNN robustness-under-scaffold-shift result in computational
chemistry) actually points to: the benefit tracks FIXED, hand-engineered
featurization vs. LEARNED, population-dependent feature extraction, not "ECFP4
specifically." A GNN's convolution weights are fit on the (held-out-or-not)
drug population; ECFP4's bits are not. If the benefit reproduces under other
FIXED featurizations that are chemically meaningful but structurally unlike
ECFP4 (MACCS keys: a different fixed bit substructure dictionary; RDKit
physicochemical descriptors: continuous, no substructure bits at all), while a
chemistry-free random control shows no effect (ruling out "any fixed input
produces this via a training-protocol artifact"), that is the clean,
mechanistically-isolated finding this section currently lacks.

Reuses pretrain()/embed_drugs()/eval_probe() from run_experiments.py verbatim
(same CLIPStyleAligner, same 25-epoch/256-batch/1e-3-lr schedule, same 3
pretraining seeds per regime, same 10-repeat drug-disjoint subsampling probe
protocol) so results are apples-to-apples with Table tab:perview's ECFP4 row
and Table tab:gnn-variants' four GNN rows. Only the structure-view input
feature matrix changes.

Writes results/fixed_featurization_replication.csv
  columns: featurization, class, full_mean, classholdout_mean, delta, p
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from run_experiments import pretrain, embed_drugs, eval_probe, CLASS_COLS, DATA, RESULTS
from splits import repeated_subsample_splits

LOGF = f"{RESULTS}/../logs/fixed_featurization_replication.log"
N_PRETRAIN_SEEDS = 3
N_REPEATS = 10


def log(msg):
    print(msg, flush=True)
    os.makedirs(os.path.dirname(LOGF), exist_ok=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


def build_maccs(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    out = np.zeros((len(smiles_list), 167), dtype=np.float32)
    n_fail = 0
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if m is None:
            n_fail += 1
            continue
        out[i] = np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.float32)
    log(f"  MACCS: {n_fail}/{len(smiles_list)} SMILES failed to parse (left as zero vector)")
    return out


def build_physchem(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    cols = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
            "RingCount", "NumAromaticRings", "FractionCSP3", "HeavyAtomCount",
            "NumValenceElectrons", "NumAliphaticRings", "NumSaturatedRings", "MolMR",
            "BalabanJ", "BertzCT", "NumHeteroatoms"]
    out = np.zeros((len(smiles_list), len(cols)), dtype=np.float64)
    n_fail = 0
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if m is None:
            n_fail += 1
            continue
        try:
            out[i] = [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
                      Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
                      Descriptors.NumRotatableBonds(m), Descriptors.RingCount(m),
                      rdMolDescriptors.CalcNumAromaticRings(m), Descriptors.FractionCSP3(m),
                      Descriptors.HeavyAtomCount(m), Descriptors.NumValenceElectrons(m),
                      Descriptors.NumAliphaticRings(m), Descriptors.NumSaturatedRings(m),
                      Descriptors.MolMR(m), Descriptors.BalabanJ(m), Descriptors.BertzCT(m),
                      rdMolDescriptors.CalcNumHeteroatoms(m)]
        except Exception:
            n_fail += 1
    log(f"  physchem: {n_fail}/{len(smiles_list)} SMILES failed (left at 0)")
    # z-score each column (fixed transform, uses no class/indication label)
    mu, sd = out.mean(axis=0), out.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return ((out - mu) / sd).astype(np.float32)


def build_random(n_drugs, dim=128, seed=777):
    """Chemistry-free null control: i.i.d. N(0,1) per drug, fixed and seeded,
    no information about real structure at all. If this ALSO showed a
    class-holdout benefit, that would mean the benefit is a training/eval
    protocol artifact independent of chemistry; if it does not (expected),
    that rules the artifact explanation out."""
    rng = np.random.RandomState(seed)
    return rng.standard_normal((n_drugs, dim)).astype(np.float32)


def run_one_featurization(name, feat, expr, meta, drug_uid_of_sig, sig_indices_all,
                           class_labels_by_drug, all_drug_uids, t0):
    rows = []
    full_embeds = []
    for pseed in range(N_PRETRAIN_SEEDS):
        m, _ = pretrain(sig_indices_all, drug_uid_of_sig, feat, expr, seed=pseed,
                         log_tag=f"{name}:full:{pseed}")
        z_s, z_e, has_e = embed_drugs(m, all_drug_uids, feat, expr, drug_uid_of_sig, sig_indices_all)
        assert has_e.all(), "BUG: zero-vector embedding artifact"
        full_embeds.append(z_s)
    log(f"[{name}] full pretraining done ({time.time()-t0:.0f}s elapsed)")

    classholdout_embeds = {cls: [] for cls in CLASS_COLS}
    for cls in CLASS_COLS:
        pos_drugs = class_labels_by_drug[cls]
        pos_drugs_set = set(pos_drugs[pos_drugs == 1].index.tolist())
        keep_mask = ~np.isin(drug_uid_of_sig, list(pos_drugs_set))
        sig_idx_keep = sig_indices_all[keep_mask]
        for pseed in range(N_PRETRAIN_SEEDS):
            m, _ = pretrain(sig_idx_keep, drug_uid_of_sig, feat, expr, seed=2000 + pseed,
                             log_tag=f"{name}:holdout:{cls}:{pseed}")
            z_s, z_e, has_e = embed_drugs(m, all_drug_uids, feat, expr, drug_uid_of_sig, sig_indices_all)
            assert has_e.all(), "BUG: zero-vector embedding artifact"
            classholdout_embeds[cls].append(z_s)
        log(f"[{name}] class-holdout {cls} done ({time.time()-t0:.0f}s elapsed)")

    for cls in CLASS_COLS:
        pos_series = class_labels_by_drug[cls]
        pos_drugs = np.array(sorted(pos_series[pos_series == 1].index.tolist()))
        pos_drugs = pos_drugs[np.isin(pos_drugs, all_drug_uids)]
        if len(pos_drugs) < 3:
            continue
        splits = repeated_subsample_splits(pos_drugs, all_drug_uids, n_repeats=N_REPEATS, seed0=42)

        full_per_split, cold_per_split = [], []
        for split in splits:
            seen, test = split["seen_drugs"], split["test_drugs"]
            test_y = split["test_labels"]
            seen_y = pos_series.reindex(seen).fillna(0).values

            full_vals = []
            for Z in full_embeds:
                r = eval_probe(Z[seen], seen_y, Z[test], test_y, probe="logreg")
                if r:
                    full_vals.append(r["auroc"])
            cold_vals = []
            for Z in classholdout_embeds[cls]:
                r = eval_probe(Z[seen], seen_y, Z[test], test_y, probe="logreg")
                if r:
                    cold_vals.append(r["auroc"])
            if full_vals and cold_vals:
                full_per_split.append(np.mean(full_vals))
                cold_per_split.append(np.mean(cold_vals))

        full_arr, cold_arr = np.array(full_per_split), np.array(cold_per_split)
        if len(full_arr) == len(cold_arr) and len(full_arr) > 1 and not np.allclose(full_arr, cold_arr):
            try:
                _, p = wilcoxon(full_arr, cold_arr)
            except ValueError:
                p = float("nan")
        else:
            p = float("nan")
        short = cls.replace("class_", "")
        rows.append(dict(featurization=name, cls=short, full_mean=full_arr.mean(),
                          classholdout_mean=cold_arr.mean(),
                          delta=cold_arr.mean() - full_arr.mean(), p=p))
        log(f"[{name}] {short}: full={full_arr.mean():.4f} cold={cold_arr.mean():.4f} "
            f"delta={cold_arr.mean()-full_arr.mean():+.4f} p={p:.4f} ({time.time()-t0:.0f}s elapsed)")
    return rows


def main():
    t0 = time.time()
    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    expr = np.load(f"{DATA}/expr_signatures.npy")
    drug_feat = pd.read_parquet(f"{DATA}/drug_features.parquet")
    n_drugs = len(drug_feat)
    assert (drug_feat["drug_uid"].values == np.arange(n_drugs)).all(), \
        "drug_uid must be a 0..n_drugs-1 row index for feature-matrix alignment"
    smiles_list = drug_feat["smiles"].tolist()

    drug_uid_of_sig = meta["drug_uid"].values.astype(int)
    sig_indices_all = np.arange(len(meta))
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    class_labels_by_drug = {cls: drug_level[cls] for cls in CLASS_COLS}
    all_drug_uids = np.array(sorted(drug_level.index.tolist()))
    log(f"n_drugs={n_drugs}, n_sigs={len(meta)}")

    featurizations = {
        "maccs": build_maccs(smiles_list),
        "physchem": build_physchem(smiles_list),
        "random_control": build_random(n_drugs),
    }
    for name, feat in featurizations.items():
        log(f"{name}: shape={feat.shape}")

    all_rows = []
    for name, feat in featurizations.items():
        all_rows += run_one_featurization(name, feat, expr, meta, drug_uid_of_sig, sig_indices_all,
                                           class_labels_by_drug, all_drug_uids, t0)
        pd.DataFrame(all_rows).to_csv(f"{RESULTS}/fixed_featurization_replication.csv", index=False)

    log(f"\nWrote fixed_featurization_replication.csv, DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
