"""Reviewer W4: an alternative, published pretrained molecular representation
alongside raw ECFP4, to check whether the "plain fingerprint beats contrastive
pretraining" finding is an artifact of comparing against a weak structural
featurization rather than a genuinely strong one.

Encodes every compound's SMILES with ChemBERTa (seyonec/ChemBERTa-zinc-base-v1,
RoBERTa pretrained on ~100M ZINC15 SMILES via masked-language modeling), mean-
pools the last hidden state over non-padding tokens, and runs the identical
random-forest baseline protocol used for raw ECFP4 in run_experiments.py,
across the same repeated drug-disjoint subsampling splits. No contrastive
pretraining is involved; this is a frozen-encoder-plus-classifier baseline,
directly comparable to the "ECFP4+RF" row in Table 1.

Writes results/chemberta_baseline.csv.
"""
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, ".")
from splits import repeated_subsample_splits

DATA = "../data"
RESULTS = "../results"
CLASS_COLS = ["class_AED", "class_Statin", "class_SSRI", "class_NSAID",
              "class_BetaBlocker", "class_Antipsychotic"]
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def embed_smiles(smiles_list, batch_size=64):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    embeds = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
            out = model(**enc).last_hidden_state  # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            embeds.append(pooled.cpu().numpy())
    return np.concatenate(embeds, axis=0)


def raw_feature_baseline(train_X, train_y, test_X, test_y):
    if len(np.unique(train_y)) < 2:
        return None
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=0)
    clf.fit(train_X, train_y)
    score = clf.predict_proba(test_X)[:, 1]
    try:
        return {"auroc": roc_auc_score(test_y, score), "auprc": average_precision_score(test_y, score)}
    except ValueError:
        return None


def main():
    drug_feat = pd.read_parquet(f"{DATA}/drug_features.parquet").sort_values("drug_uid").reset_index(drop=True)
    assert (drug_feat["drug_uid"].values == np.arange(len(drug_feat))).all(), "drug_uid must be 0..N-1 contiguous"
    meta = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    drug_level = meta.drop_duplicates("drug_uid").set_index("drug_uid")
    all_drug_uids = np.array(sorted(drug_level.index.tolist()))

    print(f"Embedding {len(drug_feat)} SMILES with {MODEL_NAME} on {DEVICE}...", flush=True)
    chemberta_X = embed_smiles(drug_feat["smiles"].tolist())
    np.save(f"{RESULTS}/chemberta_embeddings.npy", chemberta_X)
    print(f"  embeddings shape: {chemberta_X.shape}", flush=True)

    rows = []
    for cls in CLASS_COLS:
        pos_series = drug_level[cls]
        pos_drugs = np.array(sorted(pos_series[pos_series == 1].index.tolist()))
        pos_drugs = pos_drugs[np.isin(pos_drugs, all_drug_uids)]
        if len(pos_drugs) < 3:
            continue
        splits = repeated_subsample_splits(pos_drugs, all_drug_uids, n_repeats=10, seed0=42)
        for split in splits:
            seen, test, test_y = split["seen_drugs"], split["test_drugs"], split["test_labels"]
            seen_y = pos_series.reindex(seen).fillna(0).values
            r = raw_feature_baseline(chemberta_X[seen], seen_y, chemberta_X[test], test_y)
            if r:
                rows.append({"class": cls, "seed": split["seed"], "method": "chemberta_RF", **r})
        print(f"  {cls}: done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS}/chemberta_baseline.csv", index=False)
    summary = out.groupby("class").auroc.agg(["mean", "std", "count"])
    print(summary)
    print(f"\nWrote {RESULTS}/chemberta_baseline.csv")


if __name__ == "__main__":
    main()
