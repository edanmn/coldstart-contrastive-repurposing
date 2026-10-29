"""Follow-up (round 8): does the ECFP4-vs-GCN embedding-geometry divergence
(Section 4.7's GCN within-class cosine falling under class-holdout, ECFP4's
rising) trace to whether the structure FEATURE EXTRACTOR itself is retrained
under class-holdout, rather than to an unspecified "inductive bias" or
"optimization dynamics"?

ECFP4 is a fixed, dataset-independent fingerprint (RDKit hashing): only the
small adaptive MLP head on top of it needs to refit when a class is removed
from pretraining. The GCN's graph-convolution filters, by contrast, are
trained end-to-end on the contrastive objective, so under class-holdout the
entire feature extractor refits to a different training distribution, not
just a projection head.

Test: train the GCN once on full data (seed 7000, matches
gnn_embedding_diagnostics.py). For each class, build a class-holdout model
whose conv1/conv2/conv3 are INITIALIZED FROM and FROZEN AT the full-trained
weights, mimicking ECFP4's fixed feature extractor; only the final linear
projection (plus the expression encoder, unchanged from the main experiment)
retrain on the class-excluded data. Compare this frozen-extractor
class-holdout's within-class cosine similarity to (a) full and (b) the
fully-retrained class-holdout already in results/gnn_embedding_diagnostics.csv.

If freezing the extractor flips the within-cosine direction back toward
ECFP4's (rising, not falling), the divergence is attributable to whether the
feature extractor itself is data-dependent, a specific, falsifiable claim,
not a generic inductive-bias or optimization-dynamics hand-wave. If it does
not flip, that rules out this specific mechanism too.

Writes results/gnn_frozen_extractor_diagnostics.csv.
"""
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, ".")
from gnn_architecture_replication import (
    pretrain_gnn, embed_structure_all, GraphSigDataset, gnn_collate,
    DEVICE, BATCH_SIZE, LR, PRETRAIN_EPOCHS,
)
from gnn_models import GNNCLIPStyleAligner
from gnn_embedding_diagnostics import embedding_space_stats, CLASS_COLS

DATA = "../data"
RESULTS = "../results"
LOGF = "../logs/gnn_frozen_extractor_diagnostics.log"


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


def pretrain_gnn_frozen_extractor(sig_indices, drug_uid_of_sig, graphs, expr, node_dim,
                                   frozen_state, seed, log_tag):
    """Same training loop as pretrain_gnn, but conv1/conv2/conv3 are loaded
    from frozen_state and frozen; only the final linear projection and the
    expression encoder train."""
    torch.manual_seed(seed)
    ds = GraphSigDataset(sig_indices, drug_uid_of_sig, graphs, expr)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                     num_workers=0, collate_fn=gnn_collate)
    model = GNNCLIPStyleAligner(node_dim=node_dim, expr_dim=expr.shape[1]).to(DEVICE)
    se = model.structure_encoder
    se.conv1.load_state_dict(frozen_state["conv1"])
    se.conv2.load_state_dict(frozen_state["conv2"])
    se.conv3.load_state_dict(frozen_state["conv3"])
    for conv in (se.conv1, se.conv2, se.conv3):
        for p in conv.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_train = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-5)
    model.train()
    t0 = time.time()
    losses = []
    for ep in range(PRETRAIN_EPOCHS):
        ep_losses = []
        for batch in dl:
            structure = batch["structure"].to(DEVICE)
            expression = batch["expression"].to(DEVICE)
            drug_uid = batch["drug_uid"].to(DEVICE)
            loss, _, _ = model(structure, expression, drug_uid)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())
        losses.append(float(np.mean(ep_losses)))
    dt = time.time() - t0
    log(f"  [frozen-extractor:{log_tag}] {len(ds)} sigs, {PRETRAIN_EPOCHS} epochs in {dt:.1f}s, "
        f"final loss={losses[-1]:.4f} (init={losses[0]:.4f}), "
        f"trainable={n_train} frozen={n_frozen}")
    return model


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
    frozen_state = {
        "conv1": m_full.structure_encoder.conv1.state_dict(),
        "conv2": m_full.structure_encoder.conv2.state_dict(),
        "conv3": m_full.structure_encoder.conv3.state_dict(),
    }
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
        m_cold = pretrain_gnn_frozen_extractor(
            sig_idx_keep, drug_uid_of_sig, graphs, expr, node_dim,
            frozen_state, seed=7100, log_tag=f"classholdout:{short}:0")
        z_cold = embed_structure_all(m_cold, graphs)

        wf, bf, nnf = embedding_space_stats(z_full, pos_drugs, bg_drugs)
        wc, bc, nnc = embedding_space_stats(z_cold, pos_drugs, bg_drugs)
        rows.append(dict(cls=short, n_pos=len(pos_drugs),
                          within_cos_full=wf, within_cos_cold=wc,
                          to_bg_cos_full=bf, to_bg_cos_cold=bc,
                          nn5_same_class_frac_full=nnf, nn5_same_class_frac_cold=nnc))
        log(f"{short}: within-cos full={wf:.4f} cold(frozen-extractor)={wc:.4f} | "
            f"to-bg full={bf:.4f} cold={bc:.4f} ({time.time() - t0:.0f}s elapsed)")

    pd.DataFrame(rows).to_csv(f"{RESULTS}/gnn_frozen_extractor_diagnostics.csv", index=False)
    log(f"\nWrote gnn_frozen_extractor_diagnostics.csv, DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
