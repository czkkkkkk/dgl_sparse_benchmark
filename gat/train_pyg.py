import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from utils import benchmack_pyg, load_pyg_dataset
from torch_sparse import SparseTensor
import torch.nn as nn
from torch_geometric.utils import softmax
import torch_sparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, dataset = load_pyg_dataset(args.dataset, device)

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

    def forward(self, index, n, Z):
        Z = self.dropout(Z)
        Z = self.W(Z).view(Z.shape[0], self.out_size, self.num_heads)

        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        e_l = (Z * self.a_l).sum(dim=1)
        e_r = (Z * self.a_r).sum(dim=1)
        e = e_l[index[0]] + e_r[index[1]]

        a = F.leaky_relu(e)
        A_atten = softmax(a, index[1], num_nodes=n)
        a_drop = self.dropout(A_atten)
        a_drop = a_drop.transpose(0, 1)
        Z = Z.permute((2, 0, 1))
        rst = torch_sparse.spmm(index, a_drop, n, n, Z)
        rst = rst.permute((1, 2, 0))
        return rst

class GAT(nn.Module):
    def __init__(
        self, in_size, out_size, hidden_size=8, num_heads=8, dropout=0.6
    ):
        super().__init__()

        self.in_conv = GATConv(
            in_size, hidden_size, num_heads=num_heads, dropout=dropout
        )
        self.out_conv = GATConv(
            hidden_size * num_heads, out_size, num_heads=1, dropout=dropout
        )

    def forward(self, index, n, X):
        # Flatten the head and feature dimension.
        Z = F.elu(self.in_conv(index, n, X)).flatten(1)
        # Average over the head dimension.
        Z = self.out_conv(index, n, Z).mean(-1)
        return Z


model = GAT(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

edge_index = data.edge_index

# Sort columns to apply softmax
_, sorted_indices = torch.sort(edge_index[1])
edge_index = edge_index[:, sorted_indices]

def train():
    model.train()
    optimizer.zero_grad()
    out = model(edge_index, data.num_nodes, data.x)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


benchmack_pyg(20, 3, train)