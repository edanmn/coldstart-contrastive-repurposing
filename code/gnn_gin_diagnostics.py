"""Follow-up (round 8, third mechanistic test): is the GCN's
within-class-cosine reversal under class-holdout (Section 4.7) attributable
to the message-passing operator itself (GCNConv's linear, averaging-style
aggregation), rather than to pooling or feature-extractor retraining?

Ruled out so far: whether the feature extractor retrains
(gnn_frozen_extractor_diagnostics.py: freezing made it worse, not ECFP4-like)
and mean- vs max-pooling (gnn_maxpool_diagnostics.py: mixed, not a clean flip
toward ECFP4's pattern). This test isolates the conv operator: identical
depth, hidden width, and mean-pooling as the original GCN, only GCNConv
swapped for GINConv (sum-aggregation through a 2-layer MLP per hop,
theoretically more expressive, not prone to GCN's degree-normalized linear
averaging). If GIN reproduces ECFP4's rising pattern, GCN's specific
aggregation is implicated; if GIN still diverges, the conv operator choice
is also ruled out, and the open question narrows further than any single
architectural knob tested here can explain.

One seed per regime (full + 6 class-holdout), matching
gnn_embedding_diagnostics.py's protocol for direct comparability.

Writes results/gnn_gin_diagnostics.csv.
"""
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import sys
sys.path.insert(0, ".")
from gnn_architecture_replication import (
    GraphSigDataset, gnn_collate, DEVICE, BATCH_SIZE, LR, PRETRAIN_EPOCHS,
)
from gnn_models import GNNCLIPStyleAlignerGIN
from gnn_embedding_diagnostics import embedding_space_stats, CLASS_COLS

DATA = "../data"
RESULTS = "../results"
LOGF = "../logs/gnn_gin_diagnostics.log"


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


def pretrain_gnn_gin(sig_indices, drug_uid_of_sig, graphs, expr, node_dim, seed, log_tag):
    torch.manual_seed(seed)
    ds = GraphSigDataset(sig_indices, drug_uid_of_sig, graphs, expr)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                     num_workers=0, collate_fn=gnn_collate)
    model = GNNCLIPStyleAlignerGIN(node_dim=node_dim, expr_dim=expr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
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
    log(f"  [gin:{log_tag}] {len(ds)} sigs, {PRETRAIN_EPOCHS} epochs in {dt:.1f}s, "
        f"final loss={losses[-1]:.4f} (init={losses[0]:.4f})")
    return model


@torch.no_grad()
def embed_structure_all_gin(model, graphs, chunk=256):
    model.eval()
    outs = []
    for i in range(0, len(graphs), chunk):
        b = Batch.from_data_list(graphs[i:i + chunk]).to(DEVICE)
        outs.append(model.structure_encoder(b).cpu().numpy())
    return np.concatenate(outs, axis=0)


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

    m_full = pretrain_gnn_gin(sig_indices_all, drug_uid_of_sig, graphs, expr, node_dim,
                              seed=7000, log_tag="full:0")
    z_full = embed_structure_all_gin(m_full, graphs)
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
        m_cold = pretrain_gnn_gin(sig_idx_keep, drug_uid_of_sig, graphs, expr, node_dim,
                                   seed=7100, log_tag=f"classholdout:{short}:0")
        z_cold = embed_structure_all_gin(m_cold, graphs)

        wf, bf, nnf = embedding_space_stats(z_full, pos_drugs, bg_drugs)
        wc, bc, nnc = embedding_space_stats(z_cold, pos_drugs, bg_drugs)
        rows.append(dict(cls=short, n_pos=len(pos_drugs),
                          within_cos_full=wf, within_cos_cold=wc,
                          to_bg_cos_full=bf, to_bg_cos_cold=bc,
                          nn5_same_class_frac_full=nnf, nn5_same_class_frac_cold=nnc))
        log(f"{short}: within-cos full={wf:.4f} cold(GIN)={wc:.4f} | "
            f"to-bg full={bf:.4f} cold={bc:.4f} ({time.time() - t0:.0f}s elapsed)")

    pd.DataFrame(rows).to_csv(f"{RESULTS}/gnn_gin_diagnostics.csv", index=False)
    log(f"\nWrote gnn_gin_diagnostics.csv, DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
