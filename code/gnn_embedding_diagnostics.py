"""Reviewer (W1/W3 on the GCN cross-architecture replication, round 7): the
ceiling-effect account for why the GCN does not reproduce the ECFP4
structure-view benefit does not cover NSAID (full-data GCN AUROC 0.788, not
near ceiling, yet class-holdout still makes it significantly worse). The
reviewer asks for a direct comparison of class separation, neighborhood
preservation, and positive-negative similarity before and after pretraining,
for both encoders, not just the downstream AUROC delta.

Mirrors drug_level_and_embedding_analysis.py's embedding-space check
(within_cos, to_bg_cos, nn5_same_class_frac) for the GCN structure encoder,
full vs class-holdout, one seed per regime (same protocol as that script:
RNG_SEED=777, 500-drug background sample), for direct comparability.
ECFP4's numbers are already in results/embedding_space_analysis.csv; this
writes the GCN counterpart to results/gnn_embedding_diagnostics.csv so the
two can be read side by side.

Requires results/drug_graphs.pt (build_molecular_graphs.py).
"""
import time
import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, ".")
from gnn_architecture_replication import pretrain_gnn, embed_structure_all

DATA = "../data"
RESULTS = "../results"
LOGF = "../logs/gnn_embedding_diagnostics.log"
CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]
RNG_SEED = 777
BG_SAMPLE = 500


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


def embedding_space_stats(z, pos_drugs, bg_drugs):
    pos_z = z[pos_drugs]
    sim_within = pos_z @ pos_z.T
    iu = np.triu_indices(len(pos_drugs), k=1)
    within_cos = sim_within[iu].mean() if len(pos_drugs) >= 2 else float("nan")
    bg_sample = bg_drugs[np.random.RandomState(RNG_SEED).choice(
        len(bg_drugs), size=min(BG_SAMPLE, len(bg_drugs)), replace=False)]
    to_bg_cos = (pos_z @ z[bg_sample].T).mean()

    all_sim = z @ z.T
    pos_set = set(pos_drugs.tolist())
    nn_fracs = []
    for d in pos_drugs:
        order = np.argsort(-all_sim[d])
        order = [x for x in order if x != d][:5]
        nn_fracs.append(np.mean([1.0 if o in pos_set else 0.0 for o in order]))
    return float(within_cos), float(to_bg_cos), float(np.mean(nn_fracs))


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

    m_full = pretrain_gnn(sig_indices_all, drug_uid_of_sig, graphs, expr, node_dim,
                          seed=7000, log_tag="full:0")
    z_full = embed_structure_all(m_full, graphs)
    log(f"full regime embedded ({time.time() - t0:.0f}s elapsed)")

    rows = []
    for cls in CLASS_COLS:
        short = cls.replace("class_", "")
        pos_series = drug_level[cls]
        pos_drugs_set = set(pos_series[pos_series == 1].index.tolist())
        pos_drugs = np.array(sorted(pos_drugs_set))
        pos_drugs = pos_drugs[np.isin(pos_drugs, all_drug_uids)]
        bg_drugs = np.array([d for d in all_drug_uids if d not in pos_drugs_set])

        keep_mask = ~np.isin(drug_uid_of_sig, list(pos_drugs_set))
        sig_idx_keep = sig_indices_all[keep_mask]
        m_cold = pretrain_gnn(sig_idx_keep, drug_uid_of_sig, graphs, expr, node_dim,
                              seed=7100, log_tag=f"classholdout:{short}:0")
        z_cold = embed_structure_all(m_cold, graphs)

        wf, bf, nnf = embedding_space_stats(z_full, pos_drugs, bg_drugs)
        wc, bc, nnc = embedding_space_stats(z_cold, pos_drugs, bg_drugs)
        rows.append(dict(cls=short, n_pos=len(pos_drugs),
                          within_cos_full=wf, within_cos_cold=wc,
                          to_bg_cos_full=bf, to_bg_cos_cold=bc,
                          nn5_same_class_frac_full=nnf, nn5_same_class_frac_cold=nnc))
        log(f"{short}: within-cos full={wf:.4f} cold={wc:.4f} | to-bg full={bf:.4f} cold={bc:.4f} | "
            f"NN5 same-class frac full={nnf:.3f} cold={nnc:.3f} ({time.time() - t0:.0f}s elapsed)")

    pd.DataFrame(rows).to_csv(f"{RESULTS}/gnn_embedding_diagnostics.csv", index=False)
    log(f"\nWrote gnn_embedding_diagnostics.csv, DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
