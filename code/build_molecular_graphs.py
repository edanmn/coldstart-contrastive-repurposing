"""Reviewer (cross-architecture replication): build one PyG molecular graph
per drug from its resolved SMILES, indexed by drug_uid (drug_uid == row index
in data/drug_features.parquet and data/drug_ecfp4.npy, verified directly).

Atom features (28-d): one-hot element over a fixed vocabulary covering every
element actually present in this dataset (checked below, with an explicit
"other" bucket so an unseen element never crashes), one-hot degree (0-5+),
one-hot hybridization (SP/SP2/SP3/other), formal charge, aromaticity, ring
membership, total attached H count. Bonds become undirected edges (both
directions); bond order/type is not used as an edge feature (plain GCNConv
ignores edge attributes), which is the deliberate scope limit of this
replication check: a second architecture family, not a maximal one.

Writes results/drug_graphs.pt: a list of 1725 torch_geometric.data.Data
objects, graphs[drug_uid] is that drug's graph.
"""
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data

DATA = "../data"
RESULTS = "../results"

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B", "Na", "K", "Ca"]
N_ELEM = len(ELEMENTS) + 1  # +1 for "other"
HYBRIDIZATIONS = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]
N_HYB = len(HYBRIDIZATIONS) + 1  # +1 for "other"
MAX_DEGREE = 5  # 0,1,2,3,4,5+
NODE_DIM = N_ELEM + (MAX_DEGREE + 1) + N_HYB + 3  # +formal_charge,+aromatic,+in_ring (num_h folded separately below)
NODE_DIM += 1  # total_num_h


def atom_features(atom):
    feat = np.zeros(NODE_DIM, dtype=np.float32)
    sym = atom.GetSymbol()
    off = 0
    if sym in ELEMENTS:
        feat[off + ELEMENTS.index(sym)] = 1.0
    else:
        feat[off + len(ELEMENTS)] = 1.0
    off += N_ELEM
    deg = min(atom.GetDegree(), MAX_DEGREE)
    feat[off + deg] = 1.0
    off += MAX_DEGREE + 1
    hyb = atom.GetHybridization()
    if hyb in HYBRIDIZATIONS:
        feat[off + HYBRIDIZATIONS.index(hyb)] = 1.0
    else:
        feat[off + len(HYBRIDIZATIONS)] = 1.0
    off += N_HYB
    feat[off] = float(atom.GetFormalCharge())
    feat[off + 1] = 1.0 if atom.GetIsAromatic() else 0.0
    feat[off + 2] = 1.0 if atom.IsInRing() else 0.0
    off += 3
    feat[off] = float(atom.GetTotalNumHs())
    return feat


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = np.stack([atom_features(a) for a in mol.GetAtoms()])
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    if len(src) == 0:
        # single-atom molecule (e.g. a bare ion): self-loop so GCNConv has an edge to work with
        src, dst = [0], [0]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=torch.from_numpy(x).float(), edge_index=edge_index)


def main():
    df = pd.read_parquet(f"{DATA}/drug_features.parquet")
    assert (df["drug_uid"].values == np.arange(len(df))).all(), "drug_uid must equal row index"

    # confirm the element vocabulary actually covers this dataset before trusting the "other" bucket to be rare
    seen_elements = set()
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            seen_elements.update(a.GetSymbol() for a in mol.GetAtoms())
    missing = seen_elements - set(ELEMENTS)
    print(f"Elements seen: {sorted(seen_elements)}")
    print(f"Elements falling into 'other' bucket: {sorted(missing)}")

    graphs = []
    n_failed = 0
    for i, smi in enumerate(df["smiles"]):
        g = mol_to_graph(smi)
        if g is None:
            n_failed += 1
            g = Data(x=torch.zeros((1, NODE_DIM), dtype=torch.float32),
                      edge_index=torch.tensor([[0], [0]], dtype=torch.long))
        graphs.append(g)
    print(f"Built {len(graphs)} graphs ({n_failed} failed -> single dummy-atom placeholder), node_dim={NODE_DIM}")

    torch.save(graphs, f"{RESULTS}/drug_graphs.pt")
    print(f"Wrote {RESULTS}/drug_graphs.pt")


if __name__ == "__main__":
    main()
