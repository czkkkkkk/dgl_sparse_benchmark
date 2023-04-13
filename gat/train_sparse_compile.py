"""
[Graph Attention Networks]
(https://arxiv.org/abs/1710.10903)
"""

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_dataset, train, load_args
import argparse


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
    def __init__(self, in_size, out_size, hidden_size=8, num_heads=8, dropout=0.6):
        super().__init__()

        self.in_conv = GATConv(
            in_size, hidden_size, num_heads=num_heads, dropout=dropout
        )
        self.out_conv = GATConv(
            hidden_size * num_heads, out_size, num_heads=1, dropout=dropout
        )

    def forward(self, A_hat: dglsp.SparseMatrix, X: torch.Tensor):
        # Flatten the head and feature dimension.
        Z = F.elu(self.in_conv(A_hat, X)).flatten(1)
        # Average over the head dimension.
        Z = self.out_conv(A_hat, Z).mean(-1)
        return Z


def evaluate(g, pred):
    label = g.ndata["label"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # Compute accuracy on validation/test set.
    val_acc = (pred[val_mask] == label[val_mask]).float().mean()
    test_acc = (pred[test_mask] == label[test_mask]).float().mean()
    return val_acc, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = load_args(parser)
    args = parser.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g, num_classes = load_dataset(args.dataset, dev)

    # Create the sparse adjacency matrix A.
    src, dst = g.edges()
    N = g.num_nodes()
    A = dglsp.from_coo(dst, src, shape=(N, N))
    src = dst = None
    label = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    X = g.ndata["feat"]
    g = None

    # Add self-loops.
    I = dglsp.identity(A.shape, device=dev)
    A = A + I

    # Create GAT model.
    in_size = X.shape[1]
    out_size = num_classes
    model = GAT(in_size, out_size).to(dev)
    train(args, model, label, train_mask, A, X)
