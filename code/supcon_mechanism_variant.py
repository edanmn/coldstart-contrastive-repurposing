"""Reviewer W4: a second contrastive objective, to check whether the cold-start
asymmetry and the partial-recovery pattern in the mechanism test (Table~tab:mech)
are specific to the "exclude same-class pairs from negatives" debiasing
intervention, or appear under a different, also-published way of handling
same-class pairs.

class_mode="positive" in models.py implements Khosla et al. (2020)-style
supervised contrastive learning: same-class, different-drug pairs are added to
the POSITIVE set (pulled together), rather than merely excluded from the
negative set. This is a materially different objective, not a hyperparameter
variant of the debiasing run: it actively shapes the embedding to cluster the
class, instead of just removing interference.

Same data as the "debiased" regime (all drugs, nothing removed), same 3
pretraining seeds, same downstream protocol, structure view only (the view
the paper's headline claim concerns).

Writes results/supcon_mechanism.csv.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, ".")
from models import CLIPStyleAligner
from splits import repeated_subsample_splits
from run_experiments import SigDataset, embed_drugs  # reuse exact same dataset/embedding code

DATA = "../data"
RESULTS = "../results"
LOGF = "../logs/supcon_mechanism.log"
CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]
N_PRETRAIN_SEEDS = 3
N_EPOCHS = 25
BATCH_SIZE = 256
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    print(msg, flush=True)
    with open(LOGF, "a") as f:
        f.write(msg + "\n")


def pretrain_supcon(sig_indices, drug_uid_of_sig, ecfp, expr, class_label_of_drug, seed, log_tag):
    torch.manual_seed(seed)
    ds = SigDataset(sig_indices, drug_uid_of_sig, ecfp, expr, class_label_of_drug=class_label_of_drug)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    model = CLIPStyleAligner(structure_dim=ecfp.shape[1], expr_dim=expr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    model.train()
    t0 = time.time()
    last_loss = None
    for ep in range(N_EPOCHS):
        ep_losses = []
        for batch in dl:
            structure = batch["structure"].to(DEVICE)
            expression = batch["expression"].to(DEVICE)
            drug_uid = batch["drug_uid"].to(DEVICE)
            class_label = batch["class_label"].to(DEVICE)
            loss, _, _ = model(structure, expression, drug_uid, class_label, class_mode="positive")
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())
        last_loss = float(np.mean(ep_losses))
    log(f"  [pretrain:{log_tag}] {len(ds)} sigs, {N_EPOCHS} epochs in {time.time()-t0:.1f}s, final loss={last_loss:.4f}")
    return model


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
    ecfp = np.load(f"{DATA}/drug_ecfp4.npy").astype(np.float32)
    n_drugs = ecfp.shape[0]
    drug_uid_of_sig = meta["drug_uid"].values.astype(int)
    sig_indices_all = np.arange(len(meta))
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_drug_uids = np.array(sorted(drug_level.index.tolist()))

    rows = []
    for cls in CLASS_COLS:
        pos_series = drug_level[cls]
        pos_drugs_set = set(pos_series[pos_series == 1].index.tolist())
        class_label_of_drug = np.full(n_drugs, -1, dtype=np.int64)
        for d in pos_drugs_set:
            class_label_of_drug[d] = 1

        pos_drugs = np.array(sorted(pos_drugs_set))
        pos_drugs = pos_drugs[np.isin(pos_drugs, all_drug_uids)]
        if len(pos_drugs) < 3:
            continue
        splits = repeated_subsample_splits(pos_drugs, all_drug_uids, n_repeats=10, seed0=42)

        supcon_embeds = []
        for pseed in range(N_PRETRAIN_SEEDS):
            m = pretrain_supcon(sig_indices_all, drug_uid_of_sig, ecfp, expr, class_label_of_drug,
                                 seed=4000 + pseed, log_tag=f"supcon:{cls}:{pseed}")
            z_s, z_e, has_e = embed_drugs(m, all_drug_uids, ecfp, expr, drug_uid_of_sig, sig_indices_all)
            assert has_e.all(), "BUG: zero-vector embedding artifact"
            supcon_embeds.append(z_s)

        for split in splits:
            seen, test, test_y = split["seen_drugs"], split["test_drugs"], split["test_labels"]
            seen_y = pos_series.reindex(seen).fillna(0).values
            for pseed, z_s in enumerate(supcon_embeds):
                a = eval_probe(z_s[seen], seen_y, z_s[test], test_y)
                if a is not None:
                    rows.append({"class": cls, "seed": split["seed"], "pretrain_seed": pseed,
                                 "method": "supcon_structure", "auroc": a})
        log(f"{cls} done ({time.time()-t0:.0f}s elapsed)")

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/supcon_mechanism.csv", index=False)
    log(f"\nWrote {RESULTS}/supcon_mechanism.csv, DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
