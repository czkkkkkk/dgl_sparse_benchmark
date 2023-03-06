"""
[Semi-Supervised Classification with Graph Convolutional Networks]
(https://arxiv.org/abs/1609.02907)
"""
import ScheduleProfiler

profiler = ScheduleProfiler.ScheduleProfiler()

import sys

sys.path.append(
    ".."
)  # 跳到上级目录下面（sys.path添加目录时注意是在windows还是在Linux下，windows下需要‘\\'否则会出错。）

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import load_dataset, benchmark
import argparse
import time


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
        # A_norm = A_norm + A_norm
        # X = A_norm @ self.W1(X)
        
        X = dglsp.spmm(A_norm, self.W1(X))
        X = F.relu(X)
        X = dglsp.spmm(A_norm, self.W2(X))
        return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'ogbn-products', 'ogbn-arxiv').",
    )
    parser.add_argument(
        "--compile",
        type=bool,
        default=False,
    )
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
    if not args.compile:
        benchmark(50, 3, model, label, train_mask, A_norm, X)
    else:
        model_script = torch.jit.script(model)
        print(model_script.graph)
        print(model_script.code)
        benchmark(20, 3, model_script, label, train_mask, A_norm, X)
