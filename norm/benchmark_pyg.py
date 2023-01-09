import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_sparse
from torch_sparse import SparseTensor
from utils import benchmark_fn, load_pyg_dataset
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data, dataset = load_pyg_dataset(args.dataset, device)


# Use fake normalized non-zero values
init_value = torch.ones_like(data.edge_index[0]).float()

N = data.num_nodes
adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=init_value)
diag_row = torch.arange(0, N, device=device)
diag_index = torch.stack([diag_row, diag_row])

def normalize():
    degs = adj.sum(1) ** -0.5
    ret_index, value = torch_sparse.spspmm(diag_index, degs, data.edge_index, init_value, N, N, N)
    ret_index, value = torch_sparse.spspmm(ret_index, value, diag_index, degs, N, N, N)
    return ret_index, value

benchmark_fn(20, 3, normalize)