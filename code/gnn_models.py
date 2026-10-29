"""Reviewer (cross-architecture replication): a molecular-graph structure
encoder (3-layer GCN + mean pooling) in place of the ECFP4 MLP, wired into
the identical CLIP-style contrastive objective used everywhere else in this
paper. Only the structure side changes; the expression encoder and the loss
(models.CLIPStyleAligner.forward) are reused unmodified by subclassing, so
the two architectures are trained and scored under exactly the same
contrastive objective, pretraining regime, and downstream probe.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_max_pool

import sys
sys.path.insert(0, ".")
from models import CLIPStyleAligner


class GraphStructureEncoder(nn.Module):
    def __init__(self, node_dim, hidden=128, out_dim=256, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        z = self.lin(x)
        return F.normalize(z, dim=-1)


class GNNCLIPStyleAligner(CLIPStyleAligner):
    """Same contrastive objective as CLIPStyleAligner (forward/encode
    inherited unmodified); structure_encoder is replaced with a GNN that
    consumes a torch_geometric.data.Batch instead of an ECFP tensor."""

    def __init__(self, node_dim, expr_dim=978, embed_dim=256, dropout=0.2,
                 init_temp=0.07, learnable_temp=True):
        super().__init__(structure_dim=1, expr_dim=expr_dim, embed_dim=embed_dim,
                          dropout=dropout, init_temp=init_temp, learnable_temp=learnable_temp)
        self.structure_encoder = GraphStructureEncoder(node_dim, 128, embed_dim, dropout)


class GraphStructureEncoderMaxPool(nn.Module):
    """Follow-up (round 8, pooling-choice test): identical to
    GraphStructureEncoder (same conv layers, same depth, same dims) except
    global_mean_pool is replaced with global_max_pool, to test whether the
    GCN's within-class-cosine reversal under class-holdout (Section 4.7) is
    attributable to mean-pooling specifically rather than to the
    message-passing (conv) step."""

    def __init__(self, node_dim, hidden=128, out_dim=256, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = global_max_pool(x, batch)
        z = self.lin(x)
        return F.normalize(z, dim=-1)


class GNNCLIPStyleAlignerMaxPool(CLIPStyleAligner):
    """Same as GNNCLIPStyleAligner but with GraphStructureEncoderMaxPool."""

    def __init__(self, node_dim, expr_dim=978, embed_dim=256, dropout=0.2,
                 init_temp=0.07, learnable_temp=True):
        super().__init__(structure_dim=1, expr_dim=expr_dim, embed_dim=embed_dim,
                          dropout=dropout, init_temp=init_temp, learnable_temp=learnable_temp)
        self.structure_encoder = GraphStructureEncoderMaxPool(node_dim, 128, embed_dim, dropout)


class GraphStructureEncoderGIN(nn.Module):
    """Follow-up (round 8, third mechanistic test): same depth, hidden width,
    and mean-pooling as GraphStructureEncoder, but GCNConv is replaced with
    GINConv (sum-aggregation over an MLP, theoretically more expressive than
    GCN's linear aggregation, not prone to the same averaging-driven
    smoothing). Isolates the conv/message-passing operator, not pooling
    (already tested separately in GraphStructureEncoderMaxPool)."""

    def __init__(self, node_dim, hidden=128, out_dim=256, dropout=0.2):
        super().__init__()

        def gin_mlp(in_dim, out_dim_):
            return nn.Sequential(nn.Linear(in_dim, out_dim_), nn.ReLU(), nn.Linear(out_dim_, out_dim_))

        self.conv1 = GINConv(gin_mlp(node_dim, hidden))
        self.conv2 = GINConv(gin_mlp(hidden, hidden))
        self.conv3 = GINConv(gin_mlp(hidden, hidden))
        self.lin = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        z = self.lin(x)
        return F.normalize(z, dim=-1)


class GNNCLIPStyleAlignerGIN(CLIPStyleAligner):
    """Same as GNNCLIPStyleAligner but with GraphStructureEncoderGIN."""

    def __init__(self, node_dim, expr_dim=978, embed_dim=256, dropout=0.2,
                 init_temp=0.07, learnable_temp=True):
        super().__init__(structure_dim=1, expr_dim=expr_dim, embed_dim=embed_dim,
                          dropout=dropout, init_temp=init_temp, learnable_temp=learnable_temp)
        self.structure_encoder = GraphStructureEncoderGIN(node_dim, 128, embed_dim, dropout)
