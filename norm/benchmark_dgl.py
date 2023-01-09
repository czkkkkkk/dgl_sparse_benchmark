import argparse
import dgl.function as fn
import torch
from utils import load_dataset, benchmark_fn

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="cora",
    help="Dataset name ('cora', 'ogbn-products', 'ogbn-arxiv').",
)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g, _ = load_dataset(args.dataset, device)
edge_weight = torch.ones(g.number_of_edges(), device=device).float()


def normalize():
    with g.local_scope():
        g.edata["e"] = edge_weight
        g.update_all(fn.copy_e("e", "m"), fn.sum("m", "out_weight"))
        degs = g.ndata["out_weight"]
        norm = torch.pow(degs, -0.5)
        g.ndata["norm"] = norm
        g.apply_edges(lambda e: {"out": e.src["norm"] * e.dst["norm"] * e.data["e"]})
        out = g.edata["out"]
    return out


benchmark_fn(20, 3, normalize)
