import argparse
import dgl.function as fn
import torch
from utils import load_dataset, benchmark_fn
import dgl.sparse as dglsp

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

src, dst = g.edges()
N = g.num_nodes()
g = None
A = dglsp.from_coo(dst, src, shape=(N, N))
src = dst = None

I = dglsp.identity(A.shape, device=device)
A = A + I

def normalize():
    D = dglsp.diag(A.sum(1)) ** -0.5
    A_p = D @ A @ D
    return A_p


benchmark_fn(20, 3, normalize)

