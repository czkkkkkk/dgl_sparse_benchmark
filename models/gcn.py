"""
[Semi-Supervised Classification with Graph Convolutional Networks]
(https://arxiv.org/abs/1609.02907)
"""

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=16, num_layers=2):
        super().__init__()

        self.num_layers = num_layers

        # Two-layer GCN.
        self.weights = nn.ModuleList()
        self.weights.append(nn.Linear(in_size, hidden_size))
        for _ in range(self.num_layers - 2):
            self.weights.append(nn.Linear(hidden_size, hidden_size))
        self.out_conv = nn.Linear(hidden_size, out_size)

    ############################################################################
    # (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement the GCN
    # forward process.
    ############################################################################
    def forward(self, A_norm: dglsp.SparseMatrix, X: torch.Tensor):
        for _, weight in enumerate(self.weights):
            X = dglsp.spmm(A_norm, weight(X))
            X = F.relu(X)
        X = dglsp.spmm(A_norm, self.out_conv(X))
        return X
