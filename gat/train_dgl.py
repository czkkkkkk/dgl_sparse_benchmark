import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.nn as dglnn
import dgl.function as fn
from dgl.ops import edge_softmax
import argparse
from utils import load_dataset, benchmark


class GATConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, dropout=0.6):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, graph, feat):
        with graph.local_scope():
            src_prefix_shape = feat.shape[:-1]
            h_src = self.dropout(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                *src_prefix_shape, self._num_heads, self._out_feats
            )
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = F.leaky_relu(graph.edata.pop("e"))
            # compute softmax
            graph.edata["a"] = self.dropout(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            return rst


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.in_conv = GATConv(
            in_size,
            hid_size,
            heads[0],
        )
        self.out_conv = GATConv(
            hid_size * heads[0],
            out_size,
            heads[1],
        )

    def forward(self, g, X):
        Z = F.elu(self.in_conv(g, X)).flatten(1)
        Z = self.out_conv(g, Z).mean(1)
        return Z


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

    g = g.int().to(dev)
    features = g.ndata["feat"]
    labels = g.ndata["label"]

    # create GAT model
    in_size = features.shape[1]
    out_size = num_classes
    model = GAT(in_size, 8, out_size, heads=[8, 1]).to(dev)

    benchmark(20, 3, model, labels, g.ndata["train_mask"], g, features)
