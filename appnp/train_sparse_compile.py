"""
[Predict then Propagate: Graph Neural Networks meet Personalized PageRank]
(https://arxiv.org/abs/1810.05997)
"""

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_dataset, train, load_args
import argparse


class APPNP(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size=64,
        dropout=0.1,
        num_hops=10,
        alpha=0.1,
    ):
        super().__init__()

        self.f_theta = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_size),
        )
        self.num_hops = num_hops
        self.A_dropout = nn.Dropout(dropout)
        self.alpha = alpha

    def forward(self, A_hat: dglsp.SparseMatrix, X: torch.Tensor):
        Z_0 = Z = self.f_theta(X)
        for _ in range(self.num_hops):
            A_drop = dglsp.val_like(A_hat, self.A_dropout(A_hat.val))
            Z = (1 - self.alpha) * dglsp.spmm(A_drop, Z) + self.alpha * Z_0
        return Z


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = load_args(parser)
    args = parser.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g, num_classes = load_dataset(args.dataset, dev)
    label = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    X = g.ndata["feat"]

    # Create the sparse adjacency matrix A.
    src, dst = g.edges()
    N = g.num_nodes()
    g = None
    A = dglsp.from_coo(dst, src, shape=(N, N))
    src = dst = None

    # Calculate the symmetrically normalized adjacency matrix.
    I = dglsp.identity(A.shape, device=dev)
    A = A + I
    D_hat = dglsp.diag(A.sum(dim=1)) ** -0.5
    A = D_hat @ A @ D_hat
    # Create APPNP model.
    in_size = X.shape[1]
    out_size = num_classes
    model = APPNP(in_size, out_size).to(dev)

    # Kick off training.
    train(args, model, label, train_mask, A, X)
