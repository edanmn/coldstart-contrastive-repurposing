"""Extended membership-matched control (v2).

Arms, for each class C, with M a fixed random set of non-class drugs used as
probe negatives and M2 a second, disjoint random set of the same size:
  full      : nothing removed                 (pos seen,   neg seen)
  hold      : C removed                       (pos unseen, neg seen)
  sizehold  : C and M2 removed                (pos unseen, neg seen,   same data size as matched)
  matched   : C and M removed                 (pos unseen, neg unseen)
sizehold vs matched differ only in whether the probe negatives were seen, at
equal training-set size. matched vs full is the membership-free class-holdout gain.

Inputs: ecfp4 | random | chemberta | gcn | finlayson (finlayson_model.py).  Datasets: p2 (GSE70138) | p1 (GSE92742 slice).
Saves per-split AUROCs, a summary, the membership-probe AUROCs, and structure
embeddings of every arm (for drug-level LOPO without retraining).

Usage: python3 membership_control_v2.py <input> <dataset>
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from run_experiments import pretrain, eval_probe, CLASS_COLS, DATA, RESULTS, DEVICE
from fixed_featurization_replication import build_random

N_PRETRAIN_SEEDS = 3
N_REPEATS = 10
TEST_NEG = 150
TEST_POS_FRAC = 0.35
M_SIZE = {"p2": 400, "p1": 250}
EMB_DIR = f"{RESULTS}/membership_v2_embeddings"
os.makedirs(EMB_DIR, exist_ok=True)


def load_dataset(name):
    meta2 = pd.read_parquet(f"{DATA}/dataset_meta.parquet")
    drug_level = meta2.drop_duplicates("drug_uid").set_index("drug_uid")
    if name == "p2":
        expr = np.load(f"{DATA}/expr_signatures.npy")
        uid = meta2["drug_uid"].values.astype(int)
        return expr, uid, drug_level
    meta1 = pd.read_parquet(f"{DATA}/p1_meta.parquet")
    expr = np.load(f"{DATA}/p1_expr.npy").astype(np.float32)
    name2uid = {n.lower(): u for u, n in drug_level["pert_iname"].items()}
    uid = meta1["pert_iname"].str.lower().map(name2uid)
    keep = uid.notna().values
    return expr[keep], uid[keep].values.astype(int), drug_level


def build_input(name, n_drugs):
    if name == "ecfp4":
        return np.load(f"{DATA}/drug_ecfp4.npy").astype(np.float32)
    if name == "random":
        return build_random(n_drugs)
    if name == "chemberta":
        x = np.load(f"{RESULTS}/chemberta_embeddings.npy").astype(np.float32)
        return ((x - x.mean(0)) / (x.std(0) + 1e-8)).astype(np.float32)
    if name == "gcn":
        return torch.load(f"{RESULTS}/drug_graphs.pt", weights_only=False)
    if name == "finlayson":
        from finlayson_model import build_molgraphs
        return build_molgraphs(DATA)
    raise ValueError(name)


def make_trainer(inp_name, feat, expr, uid_of_sig):
    sig_all = np.arange(len(uid_of_sig))

    if inp_name == "gcn":
        import gnn_architecture_replication as g
        g.LOGF = os.path.join(os.path.dirname(__file__), "..", "logs", "membership_v2_gcn.log")
        pretrain_gnn, embed_structure_all = g.pretrain_gnn, g.embed_structure_all
        node_dim = feat[0].x.shape[1]

        def train(exclude, seed, tag):
            keep = sig_all[~np.isin(uid_of_sig, list(exclude))]
            m = pretrain_gnn(keep, uid_of_sig, feat, expr, node_dim, seed=seed, log_tag=tag)
            return embed_structure_all(m, feat)
        return train

    if inp_name == "finlayson":
        from finlayson_model import pretrain_finlayson, embed_all

        def train(exclude, seed, tag):
            keep = sig_all[~np.isin(uid_of_sig, list(exclude))]
            m = pretrain_finlayson(keep, uid_of_sig, feat, expr, seed=seed, log_tag=tag)
            return embed_all(m, len(feat))
        return train

    def train(exclude, seed, tag):
        keep = sig_all[~np.isin(uid_of_sig, list(exclude))]
        m, _ = pretrain(keep, uid_of_sig, feat, expr, seed=seed, log_tag=tag)
        m.eval()
        with torch.no_grad():
            return m.structure_encoder(torch.from_numpy(feat).float().to(DEVICE)).cpu().numpy()
    return train


def probe_splits(pos, M, seed0=42):
    out = []
    for r in range(N_REPEATS):
        rng = np.random.RandomState(seed0 + r)
        n_tp = min(max(2, int(round(len(pos) * TEST_POS_FRAC))), len(pos) - 1)
        tp = rng.choice(pos, n_tp, replace=False)
        tn = rng.choice(M, TEST_NEG, replace=False)
        trp, trn = np.setdiff1d(pos, tp), np.setdiff1d(M, tn)
        out.append((np.concatenate([trp, trn]), np.r_[np.ones(len(trp)), np.zeros(len(trn))],
                    np.concatenate([tp, tn]), np.r_[np.ones(len(tp)), np.zeros(len(tn))]))
    return out


def membership_splits(M, seen_pool, seed0=42):
    out = []
    n_te = min(TEST_NEG, len(M) // 2)
    for r in range(N_REPEATS):
        rng = np.random.RandomState(seed0 + r)
        tp = rng.choice(M, n_te, replace=False)
        tn = rng.choice(seen_pool, n_te, replace=False)
        trp = np.setdiff1d(M, tp)
        trn = rng.choice(np.setdiff1d(seen_pool, tn), len(trp), replace=False)
        out.append((np.concatenate([trp, trn]), np.r_[np.ones(len(trp)), np.zeros(len(trn))],
                    np.concatenate([tp, tn]), np.r_[np.ones(len(tp)), np.zeros(len(tn))]))
    return out


def score(Zs, splits):
    return np.array([np.mean([eval_probe(Z[a], b, Z[c], d)["auroc"] for Z in Zs])
                     for a, b, c, d in splits])


def pval(a, b):
    return float("nan") if np.allclose(a, b) else float(wilcoxon(a, b).pvalue)


def main():
    global N_PRETRAIN_SEEDS
    inp_name, ds_name = sys.argv[1], sys.argv[2]
    # optional: --seeds N  --arms hold,matched  (deadline-trimmed runs; disclosed in the paper)
    argv = sys.argv[3:]
    if "--seeds" in argv:
        N_PRETRAIN_SEEDS = int(argv[argv.index("--seeds") + 1])
    want = ["hold", "sizehold", "matched"]
    if "--arms" in argv:
        want = argv[argv.index("--arms") + 1].split(",")
    tag0 = f"{inp_name}_{ds_name}"
    t0 = time.time()
    expr, uid_of_sig, drug_level = load_dataset(ds_name)
    n_drugs = len(np.load(f"{DATA}/drug_ecfp4.npy", mmap_mode="r"))
    feat = build_input(inp_name, n_drugs)
    train = make_trainer(inp_name, feat, expr, uid_of_sig)
    universe = np.array(sorted(set(uid_of_sig.tolist())))
    print(f"[{tag0}] {len(uid_of_sig)} sigs, {len(universe)} drugs, device={DEVICE}", flush=True)

    full = [train(set(), s, f"{tag0}:full:{s}") for s in range(N_PRETRAIN_SEEDS)]
    np.save(f"{EMB_DIR}/{tag0}_full.npy", np.stack(full).astype(np.float16))
    rows, summ = [], []
    for ci, cls in enumerate(CLASS_COLS):
        short = cls.replace("class_", "")
        lab = drug_level[cls].reindex(universe).fillna(0).values
        pos, nonclass = universe[lab == 1], universe[lab == 0]
        msz = M_SIZE[ds_name]
        # same draw as membership_matched_control.py, so full/hold/matched reproduce it on p2
        M = np.sort(np.random.RandomState(900 + ci).choice(nonclass, msz, replace=False))
        M2 = np.sort(np.random.RandomState(1900 + ci).choice(np.setdiff1d(nonclass, M), msz,
                                                             replace=False))
        seen_nonclass = np.setdiff1d(nonclass, M)
        excl = {"hold": (set(pos), 2000), "sizehold": (set(pos) | set(M2), 4000),
                "matched": (set(pos) | set(M), 3000)}
        arms = {a: [train(excl[a][0], excl[a][1] + s, f"{tag0}:{a}:{short}:{s}")
                    for s in range(N_PRETRAIN_SEEDS)] for a in want}
        np.savez_compressed(f"{EMB_DIR}/{tag0}_{short}.npz", pos=pos, M=M, M2=M2,
                            **{k: np.stack(v).astype(np.float16) for k, v in arms.items()})
        sp = probe_splits(pos, M)
        sc = {"full": score(full, sp)}
        sc.update({k: score(v, sp) for k, v in arms.items()})
        msp = membership_splits(M, np.setdiff1d(seen_nonclass, M2))
        mem = {k: score(v, msp) for k, v in [("full", full), ("hold", arms["hold"]),
                                              ("matched", arms["matched"])]}
        for r in range(N_REPEATS):
            rows.append(dict(input=inp_name, dataset=ds_name, cls=short, split=r,
                             **{k: v[r] for k, v in sc.items()},
                             **{f"mem_{k}": v[r] for k, v in mem.items()}))
        rec = dict(input=inp_name, dataset=ds_name, cls=short, n_pos=len(pos),
                   n_seeds=N_PRETRAIN_SEEDS, **{k: v.mean() for k, v in sc.items()},
                   **{f"mem_{k}": v.mean() for k, v in mem.items()})
        for a in want:
            rec[f"{a}_minus_full"] = sc[a].mean() - sc["full"].mean()
            rec[f"p_{a}_vs_full"] = pval(sc[a], sc["full"])
        if "sizehold" in want:
            rec["p_sizehold_vs_matched"] = pval(sc["sizehold"], sc["matched"])
        rec["p_matched_vs_hold"] = pval(sc["matched"], sc["hold"])
        summ.append(rec)
        print(f"[{tag0}] {short}: " + " ".join(f"{k}={v.mean():.3f}" for k, v in sc.items())
              + f" mem_matched={mem['matched'].mean():.3f} {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{RESULTS}/membership_v2_{tag0}.csv", index=False)
        pd.DataFrame(summ).to_csv(f"{RESULTS}/membership_v2_{tag0}_summary.csv", index=False)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
