"""
[Semi-Supervised Classification with Graph Convolutional Networks]
(https://arxiv.org/abs/1609.02907)
"""
import ScheduleProfiler

profiler = ScheduleProfiler.ScheduleProfiler()

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_dataset, train, load_args
import argparse


class GCN(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=16):
        super().__init__()

        # Two-layer GCN.
        self.W1 = nn.Linear(in_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, out_size)

    ############################################################################
    # (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement the GCN
    # forward process.
    ############################################################################
    # @torch.jit.script
    def forward(self, A_norm: dglsp.SparseMatrix, X: torch.Tensor):
        X = dglsp.spmm(A_norm, self.W1(X))
        X = F.relu(X)
        X = dglsp.spmm(A_norm, self.W2(X))
        return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = load_args(parser)
    args = parser.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g, num_classes = load_dataset(args.dataset, dev)
    label = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    X = g.ndata["feat"]

    # Create the adjacency matrix of graph.
    src, dst = g.edges()
    N = g.num_nodes()
    g = None
    A = dglsp.from_coo(dst, src, shape=(N, N))
    src = dst = None

    ############################################################################
    # (HIGHLIGHT) Compute the symmetrically normalized adjacency matrix with
    # Sparse Matrix API
    ############################################################################
    profiler.range_push("Build A_norm")
    I = dglsp.identity(A.shape, device=dev)
    A_hat = A + I
    A = I = None
    D_hat = dglsp.diag(A_hat.sum(1)) ** -0.5
    A_norm = D_hat @ A_hat
    A_hat = None
    # A_norm = A_norm @ D_hat
    D_hat = None
    profiler.range_pop()
    # Create model.
    in_size = X.shape[1]
    out_size = num_classes
    model = GCN(in_size, out_size).to(dev)
    train(args, model, label, train_mask, A_norm, X)
