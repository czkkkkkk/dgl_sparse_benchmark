# record the fw and bw time of DGL sparse
import dgl
import dgl.sparse as dglsp
import torch

from models import GAT, GCN
from utils import benchmark_fw_bw, benchmark_tensorboard

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_nodes, num_edges = 10_000, 200_000
    feature = torch.randn(num_nodes, 64, device=dev)
    src, dst = torch.randint(num_nodes, (2, num_edges), device=dev)
    A = dglsp.from_coo(src, dst, shape=(num_nodes, num_nodes))
    I = dglsp.identity(A.shape, device=dev)
    A = A + I

    for Model in [GCN]:
        print(f"Model: {Model.__name__}")

        model = Model(64, 64, num_layers=3).to(dev)
        compiled_model = torch.compile(model)

        benchmark_fw_bw(
            epochs=50 if dev == "cpu" else 500,
            warmup=10 if dev == "cpu" else 100,
            models=[model, compiled_model],
            model_names=["Vanilla", "Compiled"],
            args=(A, feature),
            backward=args.backward,
        )
        benchmark_tensorboard(
            models=[model, compiled_model],
            model_names=['DGL_Vanilla', 'DGL_Compiled'],
            args=(A, feature),
            epochs=5
        )
