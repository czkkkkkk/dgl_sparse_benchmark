"""
[Graph Attention Networks]
(https://arxiv.org/abs/1710.10903)
"""

import argparse

import dgl.sparse as dglsp
import torch

from models import GAT, GCN, SIGN
from utils import benchmark, load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'ogbn-products', 'ogbn-arxiv').",
    )
    parser.add_argument("--compile", action="store_true")
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
    model = GAT(in_size, out_size, num_layer=3).to(dev)
    if not args.compile:
        benchmark(20, 3, model, label, train_mask, A, X)
    else:
        model_script = torch.jit.script(model)
        print(model_script.graph)
        print(model_script.code)
        benchmark(20, 3, model_script, label, train_mask, A, X)
