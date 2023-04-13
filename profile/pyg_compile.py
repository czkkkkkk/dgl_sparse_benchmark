import torch
import torch.nn.functional as F

import torch_geometric.typing
from torch_geometric.nn.models import GAT, GCN, GIN, PNA, EdgeCNN, GraphSAGE
from utils import benchmark_tensorboard, benchmark_fw_bw
from torch_sparse import SparseTensor
import pdb


def compile(model):
    compiled_model = torch_geometric.compile(model)
    return compiled_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    x = torch.randn(num_nodes, 64, device=args.device)
    edge_index = torch.randint(num_nodes, (2, num_edges), device=args.device)
    adj = SparseTensor(
        row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes)
    )
    pdb.set_trace()
    # for Model in [GCN]:
    for Model in [GCN, GraphSAGE, GIN, EdgeCNN]:
        print(f"Model: {Model.__name__}")

        model = Model(64, 64, num_layers=3, cached=True).to(args.device)
        compiled_model = torch_geometric.compile(model)
        # origin_model = Model(64, 64, num_layers=3, cached= True).to(args.device)
        # print(model.convs)
        # print(compiled_model.convs)
        # print(jittable_model.convs)

        # benchmark_tensorboard(
        #     models=[model, compiled_model, origin_model],
        #     model_names=['Vanilla', 'Compiled', "Origin"],
        #     args=(x, edge_index),
        #     epochs=5
        # )
        # benchmark_fw_bw(
        #     epochs=50 if args.device == "cpu" else 500,
        #     warmup=10 if args.device == "cpu" else 100,
        #     models=[model, compiled_model, origin_model],
        #     model_names=['Vanilla', 'Compiled', "Origin"],
        #     args=(x, edge_index),
        #     backward=args.backward,
        # )
