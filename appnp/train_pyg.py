import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.logging import init_wandb, log
from utils import benchmark_fn, load_pyg_dataset
from torch_geometric.nn import APPNP
import torch.nn as nn
import torch_sparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cora")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data, dataset = load_pyg_dataset(args.dataset, device)


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

    def forward(self, index, value, n, X):
        Z_0 = Z = self.f_theta(X)
        for _ in range(self.num_hops):
            A_drop = self.A_dropout(value)
            lhs = torch_sparse.spmm(index, (1 - self.alpha) * A_drop, n, n, Z)
            rhs = self.alpha * Z_0
            Z = lhs + rhs
        return Z


in_size = data.x.shape[1]
out_size = dataset.num_classes
model = APPNP(in_size, out_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

# Use fake normalized non-zero values
value = torch.ones_like(data.edge_index[0]).float()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.edge_index, value, data.num_nodes, data.x)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


benchmark_fn(20, 3, train)
