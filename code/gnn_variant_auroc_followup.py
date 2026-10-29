"""Follow-up (round 9, reviewer W2): the three architectural-variant
ablations in Section 4.7 (frozen extractor, max-pooling, GIN) reported only
within-class cosine similarity, an embedding-geometry diagnostic, not the
paper's actual classification metric. Within-class cosine moving is not by
itself evidence that the downstream few-shot probe's AUROC moved the same
way. This computes the same downstream structure-view AUROC comparison
(class-holdout minus full, 10 subsampling splits, paired Wilcoxon) used for
the primary GCN result (gnn_architecture_replication.py), for all three
variants, one pretrain seed each (matching the embedding diagnostics'
protocol; the main result averages 3 seeds, these do not).

For the frozen-extractor variant, "full" is the standard, fully-trainable
GCN (it supplies the weights that get frozen); only class-holdout freezes
conv1-3. For max-pooling and GIN, both full and class-holdout use the same
variant-specific architecture, matching their diagnostic scripts.

Retrains each variant from scratch; embeddings/models from the earlier
diagnostic runs were not persisted to disk.

Writes results/gnn_variant_auroc_followup.csv.
"""
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

import sys
sys.path.insert(0, ".")
from gnn_architecture_replication import eval_probe, pretrain_gnn, embed_structure_all
from splits import repeated_subsample_splits
from gnn_frozen_extractor_diagnostics import pretrain_gnn_frozen_extractor
from gnn_maxpool_diagnostics import pretrain_gnn_maxpool, embed_structure_all_maxpool
from gnn_gin_diagnostics import pretrain_gnn_gin, embed_structure_all_gin

DATA = "../data"
RESULTS = "../results"
LOGF = "../logs/gnn_variant_auroc_followup.log"
CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


def downstream_eval(z_full, z_cold, pos_drugs, all_drug_uids, pos_series):
    splits = repeated_subsample_splits(pos_drugs, all_drug_uids, n_repeats=10, seed0=42)
    aurocs_full, aurocs_cold = [], []
    for split in splits:
        seen, test, test_y = split["seen_drugs"], split["test_drugs"], split["test_labels"]
        seen_y = pos_series.reindex(seen).fillna(0).values
        a_full = eval_probe(z_full[seen], seen_y, z_full[test], test_y)
        a_cold = eval_probe(z_cold[seen], seen_y, z_cold[test], test_y)
        if a_full is not None and a_cold is not None:
            aurocs_full.append(a_full)
            aurocs_cold.append(a_cold)
    af, ac = np.array(aurocs_full), np.array(aurocs_cold)
    p = wilcoxon(ac, af).pvalue if not np.allclose(ac, af) else float("nan")
    return af.mean(), ac.mean(), p


def run_classholdout_loop(name, z_full, cold_pretrain_fn, meta, expr, graphs, node_dim,
                           drug_uid_of_sig, sig_indices_all, drug_level, all_drug_uids,
                           embed_fn, t0):
    rows = []
    for cls in CLASS_COLS:
        short = cls.replace("class_", "")
        pos_series = drug_level[cls]
        pos_drugs_set = set(pos_series[pos_series == 1].index.tolist())
        pos_drugs = np.array(sorted(pos_drugs_set))
        pos_drugs = pos_drugs[np.isin(pos_drugs, all_drug_uids)]

        keep_mask = ~np.isin(drug_uid_of_sig, list(pos_drugs_set))
        sig_idx_keep = sig_indices_all[keep_mask]
        m_cold = cold_pretrain_fn(sig_idx_keep, drug_uid_of_sig, graphs, expr, node_dim,
                                   seed=7100, log_tag=f"{name}:classholdout:{short}:0")
        z_cold = embed_fn(m_cold, graphs)

        auroc_full, auroc_cold, p = downstream_eval(z_full, z_cold, pos_drugs, all_drug_uids, pos_series)
        rows.append(dict(variant=name, cls=short, auroc_full=auroc_full, auroc_cold=auroc_cold,
                          delta=auroc_cold - auroc_full, p=p))
        log(f"[{name}] {short}: auroc full={auroc_full:.3f} cold={auroc_cold:.3f} "
            f"delta={auroc_cold - auroc_full:+.3f} p={p:.4f} ({time.time() - t0:.0f}s elapsed)")
    return rows


def main():
    t0 = time.time()
    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    expr = np.load(f"{DATA}/expr_signatures.npy")
    graphs = torch.load(f"{RESULTS}/drug_graphs.pt", weights_only=False)
    node_dim = graphs[0].x.shape[1]
    drug_uid_of_sig = meta["drug_uid"].values.astype(int)
    sig_indices_all = np.arange(len(meta))
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_drug_uids = np.array(sorted(drug_level.index.tolist()))
    log(f"node_dim={node_dim}, n_drugs={len(graphs)}, n_sigs={len(meta)}")

    all_rows = []

    # --- frozen extractor: "full" is the standard, fully-trainable GCN ---
    m_full_std = pretrain_gnn(sig_indices_all, drug_uid_of_sig, graphs, expr, node_dim,
                               seed=7000, log_tag="frozen-src-full:0")
    z_full_std = embed_structure_all(m_full_std, graphs)
    frozen_state = {
        "conv1": m_full_std.structure_encoder.conv1.state_dict(),
        "conv2": m_full_std.structure_encoder.conv2.state_dict(),
        "conv3": m_full_std.structure_encoder.conv3.state_dict(),
    }
    log(f"frozen-extractor source (standard full) embedded ({time.time() - t0:.0f}s elapsed)")

    def frozen_pretrain(sig_idx, duos, g, e, nd, seed, log_tag):
        return pretrain_gnn_frozen_extractor(sig_idx, duos, g, e, nd, frozen_state, seed, log_tag)

    all_rows += run_classholdout_loop("frozen_extractor", z_full_std, frozen_pretrain,
                                       meta, expr, graphs, node_dim, drug_uid_of_sig,
                                       sig_indices_all, drug_level, all_drug_uids,
                                       embed_structure_all, t0)

    # --- max-pooling: full and class-holdout both use the max-pool architecture ---
    m_full_mp = pretrain_gnn_maxpool(sig_indices_all, drug_uid_of_sig, graphs, expr, node_dim,
                                      seed=7000, log_tag="maxpool:full:0")
    z_full_mp = embed_structure_all_maxpool(m_full_mp, graphs)
    log(f"maxpool full embedded ({time.time() - t0:.0f}s elapsed)")
    all_rows += run_classholdout_loop("maxpool", z_full_mp, pretrain_gnn_maxpool,
                                       meta, expr, graphs, node_dim, drug_uid_of_sig,
                                       sig_indices_all, drug_level, all_drug_uids,
                                       embed_structure_all_maxpool, t0)

    # --- GIN: full and class-holdout both use the GIN architecture ---
    m_full_gin = pretrain_gnn_gin(sig_indices_all, drug_uid_of_sig, graphs, expr, node_dim,
                                   seed=7000, log_tag="gin:full:0")
    z_full_gin = embed_structure_all_gin(m_full_gin, graphs)
    log(f"GIN full embedded ({time.time() - t0:.0f}s elapsed)")
    all_rows += run_classholdout_loop("gin", z_full_gin, pretrain_gnn_gin,
                                       meta, expr, graphs, node_dim, drug_uid_of_sig,
                                       sig_indices_all, drug_level, all_drug_uids,
                                       embed_structure_all_gin, t0)

    pd.DataFrame(all_rows).to_csv(f"{RESULTS}/gnn_variant_auroc_followup.csv", index=False)
    log(f"\nWrote gnn_variant_auroc_followup.csv, DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
