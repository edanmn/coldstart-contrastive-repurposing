"""Reviewer (cross-architecture replication, highest-priority requested
check): does the paper's central finding, that class-holdout pretraining
beats full-data pretraining on the structure view in five of six classes,
survive swapping the ECFP4 MLP structure encoder for a molecular-graph GCN
encoder under the identical contrastive objective, pretraining regimes, and
downstream probe?

Scope, deliberately: this reruns only the structure-view full-vs-class-holdout
comparison (Table 2's "Structure" row), the specific claim under test, not the
whole paper's grid (no random-holdout control, no debiasing variant, no
expression view). Same CLIP-style loss (models.CLIPStyleAligner.forward,
reused unmodified via subclassing in gnn_models.py), same 25 epochs/batch
256/AdamW lr 1e-3, same 3 pretraining seeds per regime, same 10-seed repeated
drug-disjoint subsampling + logistic-regression probe, same paired Wilcoxon
test, so any difference in outcome is attributable to the structure encoder
architecture and not to an uncontrolled pipeline difference.

Requires results/drug_graphs.pt (build_molecular_graphs.py).
Writes results/gnn_architecture_replication.csv.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon

import sys
sys.path.insert(0, ".")
from gnn_models import GNNCLIPStyleAligner
from splits import repeated_subsample_splits

DATA = "../data"
RESULTS = "../results"
LOGF = "../logs/gnn_architecture_replication.log"
CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SEEDS = 3
PRETRAIN_EPOCHS = 25
BATCH_SIZE = 256
LR = 1e-3


def log(msg):
    print(msg, flush=True)
    os.makedirs(os.path.dirname(LOGF), exist_ok=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


class GraphSigDataset(Dataset):
    """Mirrors run_experiments.SigDataset, but 'structure' is a graph index
    (resolved to the actual Data object only at collate time) rather than an
    ECFP tensor."""

    def __init__(self, sig_indices, drug_uid_of_sig, graphs, expr):
        self.sig_indices = sig_indices
        self.drug_uid_of_sig = drug_uid_of_sig
        self.graphs = graphs
        self.expr = expr

    def __len__(self):
        return len(self.sig_indices)

    def __getitem__(self, i):
        s = self.sig_indices[i]
        d = int(self.drug_uid_of_sig[s])
        return {
            "graph": self.graphs[d],
            "expression": torch.from_numpy(self.expr[s]).float(),
            "drug_uid": torch.tensor(d, dtype=torch.long),
        }


def gnn_collate(batch_list):
    structure = Batch.from_data_list([item["graph"] for item in batch_list])
    expression = torch.stack([item["expression"] for item in batch_list])
    drug_uid = torch.stack([item["drug_uid"] for item in batch_list])
    return {"structure": structure, "expression": expression, "drug_uid": drug_uid}


def pretrain_gnn(sig_indices, drug_uid_of_sig, graphs, expr, node_dim, seed, log_tag):
    torch.manual_seed(seed)
    ds = GraphSigDataset(sig_indices, drug_uid_of_sig, graphs, expr)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                     num_workers=0, collate_fn=gnn_collate)
    model = GNNCLIPStyleAligner(node_dim=node_dim, expr_dim=expr.shape[1]).to(DEVICE)
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
    log(f"  [pretrain:{log_tag}] {len(ds)} sigs, {PRETRAIN_EPOCHS} epochs in {dt:.1f}s, "
        f"final loss={losses[-1]:.4f} (init={losses[0]:.4f})")
    return model


@torch.no_grad()
def embed_structure_all(model, graphs, chunk=256):
    model.eval()
    outs = []
    for i in range(0, len(graphs), chunk):
        b = Batch.from_data_list(graphs[i:i + chunk]).to(DEVICE)
        outs.append(model.structure_encoder(b).cpu().numpy())
    return np.concatenate(outs, axis=0)


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
    graphs = torch.load(f"{RESULTS}/drug_graphs.pt", weights_only=False)
    node_dim = graphs[0].x.shape[1]
    drug_uid_of_sig = meta["drug_uid"].values.astype(int)
    sig_indices_all = np.arange(len(meta))
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_drug_uids = np.array(sorted(drug_level.index.tolist()))
    log(f"node_dim={node_dim}, n_drugs={len(graphs)}, n_sigs={len(meta)}, device={DEVICE}")

    # --- full regime: trained once, reused across all six classes ---
    full_z_s = []
    for seed in range(N_SEEDS):
        m = pretrain_gnn(sig_indices_all, drug_uid_of_sig, graphs, expr, node_dim,
                          seed=7000 + seed, log_tag=f"full:{seed}")
        full_z_s.append(embed_structure_all(m, graphs))
    log(f"full regime done ({time.time()-t0:.0f}s elapsed)")

    rows = []
    for cls in CLASS_COLS:
        short = cls.replace("class_", "")
        pos_series = drug_level[cls]
        pos_drugs_set = set(pos_series[pos_series == 1].index.tolist())
        pos_drugs = np.array(sorted(pos_drugs_set))
        pos_drugs = pos_drugs[np.isin(pos_drugs, all_drug_uids)]
        splits = repeated_subsample_splits(pos_drugs, all_drug_uids, n_repeats=10, seed0=42)

        cold_z_s = []
        keep_mask = ~np.isin(drug_uid_of_sig, list(pos_drugs_set))
        sig_idx_keep = sig_indices_all[keep_mask]
        for seed in range(N_SEEDS):
            m = pretrain_gnn(sig_idx_keep, drug_uid_of_sig, graphs, expr, node_dim,
                              seed=7100 + seed, log_tag=f"classholdout:{short}:{seed}")
            cold_z_s.append(embed_structure_all(m, graphs))
        log(f"{short}: class-holdout done ({time.time()-t0:.0f}s elapsed)")

        for split in splits:
            seen, test, test_y = split["seen_drugs"], split["test_drugs"], split["test_labels"]
            seen_y = pos_series.reindex(seen).fillna(0).values
            for pseed in range(N_SEEDS):
                a_full = eval_probe(full_z_s[pseed][seen], seen_y, full_z_s[pseed][test], test_y)
                a_cold = eval_probe(cold_z_s[pseed][seen], seen_y, cold_z_s[pseed][test], test_y)
                if a_full is not None:
                    rows.append({"class": short, "split_seed": split["seed"], "pretrain_seed": pseed,
                                 "regime": "full", "auroc": a_full})
                if a_cold is not None:
                    rows.append({"class": short, "split_seed": split["seed"], "pretrain_seed": pseed,
                                 "regime": "classholdout", "auroc": a_cold})

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/gnn_architecture_replication.csv", index=False)

    log("\n=== Summary (structure view, GCN encoder) ===")
    for cls in [c.replace("class_", "") for c in CLASS_COLS]:
        sub = out[out["class"] == cls]
        full_vals = sub[sub.regime == "full"].groupby("split_seed").auroc.mean()
        cold_vals = sub[sub.regime == "classholdout"].groupby("split_seed").auroc.mean()
        common = full_vals.index.intersection(cold_vals.index)
        f, c = full_vals.loc[common].values, cold_vals.loc[common].values
        delta = c.mean() - f.mean()
        p = wilcoxon(c, f).pvalue if not np.allclose(c, f) else float("nan")
        log(f"{cls}: full={f.mean():.3f} classholdout={c.mean():.3f} delta={delta:+.3f} p={p:.4f}")

    log(f"\nWrote {RESULTS}/gnn_architecture_replication.csv, DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
