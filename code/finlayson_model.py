"""Adaptation of Finlayson et al. (PSB 2021) coordinated structure-expression embeddings,
ported from their public code (github.com/sgfin/molecule_ge_coordinated_embeddings,
models.py FeedForwardTripletNet / losses.py QuadrupletMarginLoss) into our pipeline.

Kept from the published code:
  - expression encoder: self-normalizing net (SELU, AlphaDropout p=0), 978 -> 1024 -> 512 -> 128
  - structure encoder: Chemprop D-MPNN (hidden 300, depth 3) -> linear 300 -> 128
  - loss: quadruplet margin loss = triplet(anchor_ge, anchor_chem, nonmatch_chem)
          + triplet(anchor_chem, anchor_ge, nonmatch_ge); Euclidean distance,
          beta = 1.2, margin = 0.2 (TripletMarginLoss_WU defaults)
  - one random non-matching signature per anchor; Adam, lr 3e-4, batch 512
Changed (disclosed in the paper):
  - D-MPNN trained from scratch with Chemprop 2.0 (their PCBA-pretrained checkpoint is not public)
  - 25 epochs (their default 100) to match every other encoder in this study
  - non-match drawn from a different drug (their sampler draws a different perturbagen)
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from chemprop import featurizers
from chemprop import nn as cnn
from chemprop.data import BatchMolGraph
from rdkit import Chem

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, LR, BETA, MARGIN = 25, 512, 3e-4, 1.2, 0.2


def build_molgraphs(data_dir):
    smi = pd.read_parquet(f"{data_dir}/drug_features.parquet").sort_values("drug_uid")
    assert (smi.drug_uid.values == np.arange(len(smi))).all()
    fz = featurizers.SimpleMoleculeMolGraphFeaturizer()
    return [fz(Chem.MolFromSmiles(s)) for s in smi.smiles.values]


class SNN(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.SELU(), nn.AlphaDropout(0.0)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ChemEncoder(nn.Module):
    def __init__(self, molgraphs):
        super().__init__()
        self.molgraphs = molgraphs
        self.mp = cnn.BondMessagePassing(d_h=300, depth=3)
        self.agg = cnn.MeanAggregation()
        self.lin = nn.Linear(300, 128)

    def forward(self, drug_idx):
        bmg = BatchMolGraph([self.molgraphs[int(i)] for i in drug_idx])
        bmg.to(DEVICE)
        h = self.agg(self.mp(bmg), bmg.batch)
        return self.lin(h)


class FinlaysonNet(nn.Module):
    def __init__(self, molgraphs, expr_dim=978):
        super().__init__()
        self.ge = SNN([expr_dim, 1024, 512, 128])
        self.chem = ChemEncoder(molgraphs)

    def encode_drugs(self, drug_idx):
        uniq, inv = np.unique(drug_idx, return_inverse=True)
        z = self.chem(uniq)
        return z[torch.as_tensor(inv, device=z.device)]


def triplet(anchor, pos, neg):
    d_ap = (anchor - pos).norm(dim=1)
    d_an = (anchor - neg).norm(dim=1)
    pl = torch.clamp(d_ap - BETA + MARGIN, min=0)
    nl = torch.clamp(BETA - d_an + MARGIN, min=0)
    cnt = int(((pl > 0).int() + (nl > 0).int()).sum().item())
    return (pl + nl).sum() / max(cnt, 1)


def pretrain_finlayson(sig_indices, uid_of_sig, molgraphs, expr, seed, log_tag):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = FinlaysonNet(molgraphs, expr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sig_indices = np.asarray(sig_indices)
    t0, losses = time.time(), []
    model.train()
    for ep in range(EPOCHS):
        perm = rng.permutation(sig_indices)
        ep_l = []
        for b in range(0, len(perm) - BATCH + 1, BATCH):
            a = perm[b:b + BATCH]
            nm = rng.choice(sig_indices, BATCH)
            same = uid_of_sig[nm] == uid_of_sig[a]
            while same.any():
                nm[same] = rng.choice(sig_indices, int(same.sum()))
                same = uid_of_sig[nm] == uid_of_sig[a]
            ge_a = model.ge(torch.from_numpy(expr[a]).float().to(DEVICE))
            ge_n = model.ge(torch.from_numpy(expr[nm]).float().to(DEVICE))
            ch_a = model.encode_drugs(uid_of_sig[a])
            ch_n = model.encode_drugs(uid_of_sig[nm])
            loss = triplet(ge_a, ch_a, ch_n) + triplet(ch_a, ge_a, ge_n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_l.append(loss.item())
        losses.append(float(np.mean(ep_l)))
    print(f"  [pretrain:{log_tag}] {len(sig_indices)} sigs, {EPOCHS} epochs in "
          f"{time.time()-t0:.1f}s, final loss={losses[-1]:.4f} (init={losses[0]:.4f})", flush=True)
    return model


@torch.no_grad()
def embed_all(model, n_drugs, chunk=512):
    model.eval()
    out = [model.chem(np.arange(i, min(i + chunk, n_drugs))).cpu().numpy()
           for i in range(0, n_drugs, chunk)]
    return np.concatenate(out)
