import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl import AddSelfLoop
from utils import load_dataset, benchmark


# pylint: disable=W0235
class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats):
        super(GraphConv, self).__init__()
        self.W = nn.Linear(in_feats, out_feats)

    
    def forward(self, graph, feat):
        with graph.local_scope():
            aggregate_fn = fn.u_mul_e('h', 'e', 'm')

            feat = self.W(feat)
            graph.srcdata['h'] = feat
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

            rst = torch.relu(rst)

            return rst

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            GraphConv(in_size, hid_size)
        )
        self.layers.append(GraphConv(hid_size, out_size))

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        return h


def preprocess(g):
    g.edata['e'] = torch.ones(g.number_of_edges(), device=g.device, dtype=torch.float)
    g.ndata['i'] = g.in_degrees() ** -0.5
    g.apply_edges(fn.u_mul_e('i', 'e', out='e'))
    g.apply_edges(fn.v_mul_e('i', 'e', out='e'))
    return g

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'ogbn-products', 'ogbn-arxiv').",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g, num_classes = load_dataset(args.dataset, device)

    g = preprocess(g)

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # create GCN model
    in_size = features.shape[1]
    out_size = num_classes 
    model = GCN(in_size, 16, out_size).to(device)


    benchmark(20, 3, model, labels, g.ndata['train_mask'], g, features)


