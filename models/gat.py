"""
[Graph Attention Networks]
(https://arxiv.org/abs/1710.10903)
"""

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATConv(nn.Module):
    def __init__(self, in_size, out_size, num_heads, dropout):
        super().__init__()

        self.out_size = out_size
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.W = nn.Linear(in_size, out_size * num_heads)
        self.a_l = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.a_r = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.W.weight, gain=gain)
        nn.init.xavier_normal_(self.a_l, gain=gain)
        nn.init.xavier_normal_(self.a_r, gain=gain)

    ###########################################################################
    # (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement
    # multihead attention.
    ###########################################################################
    def forward(self, A_hat: dglsp.SparseMatrix, Z: torch.Tensor):
        Z = self.dropout(Z)
        Z = self.W(Z).view(Z.shape[0], self.out_size, self.num_heads)

        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        e_l = (Z * self.a_l).sum(dim=1)
        e_r = (Z * self.a_r).sum(dim=1)
        e = e_l[A_hat.row] + e_r[A_hat.col]

        a = F.leaky_relu(e)
        A_atten = dglsp.val_like(A_hat, a).softmax()
        a_drop = self.dropout(A_atten.val)
        A_atten = dglsp.val_like(A_atten, a_drop)
        rst = dglsp.bspmm(A_atten, Z)
        return rst


class GAT(nn.Module):
    def __init__(
        self, in_size, out_size, hidden_size=8, num_heads=8, dropout=0.6, num_layers=2
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            GATConv(in_size, hidden_size, num_heads=num_heads, dropout=dropout)
        )
        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(
                    hidden_size * num_heads,
                    hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )
        self.out_conv = GATConv(
            hidden_size * num_heads, out_size, num_heads=1, dropout=dropout
        )

    def forward(self, A_hat: dglsp.SparseMatrix, X: torch.Tensor):
        # Flatten the head and feature dimension.
        for _, layer in enumerate(self.layers):
            X = F.elu(layer(A_hat, X)).flatten(1)
        # Average over the head dimension.
        X = self.out_conv(A_hat, X).mean(-1)
        return X
