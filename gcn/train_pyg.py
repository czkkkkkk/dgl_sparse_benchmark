import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_sparse
from utils import benchmark_fn, load_pyg_dataset
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cora")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data, dataset = load_pyg_dataset(args.dataset, device)


class GCN(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=16):
        super().__init__()

        # Two-layer GCN.
        self.W1 = nn.Linear(in_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, out_size)

    def forward(self, index, value, n, m, X):
        X = self.W1(X)
        X = torch_sparse.spmm(index, value, n, m, X)
        X = F.relu(X)
        X = self.W2(X)
        X = torch_sparse.spmm(index, value, n, m, X)
        X = F.relu(X)
        return X


model = GCN(dataset.num_features, dataset.num_classes)
model, data = model.to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)


# Use fake normalized non-zero values
value = torch.ones_like(data.edge_index[0]).float()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.edge_index, value, data.num_nodes, data.num_nodes, data.x)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


benchmark_fn(20, 3, train)
