import argparse
import time

import numpy as np
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_dataset, benchmark

import dgl


class APPNPConv(nn.Module):
    def __init__(self, k, alpha, edge_drop=0.0):
        super(APPNPConv, self).__init__()
        self._k = k
        self._alpha = alpha
        self.edge_drop = nn.Dropout(edge_drop)

    def forward(self, graph, feat, edge_weight):
        with graph.local_scope():
            feat_0 = feat
            for _ in range(self._k):
                graph.ndata["h"] = feat
                graph.edata["w"] = self.edge_drop(edge_weight).to(feat.device)
                graph.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"))
                feat = graph.ndata.pop("h")
                feat = (1 - self._alpha) * feat + self._alpha * feat_0
            return feat


class APPNP(nn.Module):
    def __init__(
        self,
        g,
        in_feats,
        hidden,
        n_classes,
        activation,
        feat_drop,
        edge_drop,
        alpha,
        k,
    ):
        super(APPNP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hidden))
        self.layers.append(nn.Linear(hidden, n_classes))
        self.activation = activation
        self.feat_drop = nn.Dropout(feat_drop)
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features, edge_weight):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        h = self.feat_drop(h)
        h = self.layers[1](h)
        h = self.propagate(self.g, h, edge_weight)
        return h


def preprocess(g):
    g.edata['e'] = torch.ones(g.number_of_edges(), device=g.device, dtype=torch.float)
    g.ndata['i'] = g.in_degrees() ** -0.5
    g.apply_edges(fn.u_mul_e('i', 'e', out='e'))
    g.apply_edges(fn.v_mul_e('i', 'e', out='e'))
    return g

def main(args, g, num_classes):

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    in_feats = features.shape[1]
    n_classes = num_classes

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = preprocess(g)

    # create APPNP model
    model = APPNP(
        g,
        in_feats,
        64,
        n_classes,
        F.relu,
        0.1,
        0,
        0.1,
        10,
    )

    model.cuda()
    benchmark(20, 3, model, labels, train_mask, features, g.edata['e'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'ogbn-products', 'ogbn-arxiv').",
    )
    args = parser.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g, num_classes = load_dataset(args.dataset, dev)
    main(args, g, num_classes)
