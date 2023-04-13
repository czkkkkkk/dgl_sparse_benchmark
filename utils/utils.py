import os.path as osp
import time
from typing import Any, Callable, List, Optional, Tuple

import dgl
import torch
import torch.nn as nn
from dgl.data import CoraGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset
from pynvml import *
from torch.optim import Adam

nvmlInit()


def OgbDataset(graph_name, dev):
    assert graph_name in ("ogbn-products", "ogbn-arxiv")
    dataset = DglNodePropPredDataset(name=graph_name, root="./dataset")
    g, labels = dataset[0]
    g = dgl.add_self_loop(g).to(dev)
    split_idx = dataset.get_idx_split()
    # get split labels
    g.ndata["label"] = labels.to(dev).view(-1)
    g.ndata["train_mask"] = torch.zeros(
        g.number_of_nodes(), dtype=torch.bool, device=dev
    )
    g.ndata["train_mask"][split_idx["train"]] = True
    g.ndata["val_mask"] = torch.zeros(g.number_of_nodes(), dtype=torch.bool, device=dev)
    g.ndata["val_mask"][split_idx["valid"]] = True
    g.ndata["test_mask"] = torch.zeros(
        g.number_of_nodes(), dtype=torch.bool, device=dev
    )
    g.ndata["test_mask"][split_idx["test"]] = True
    num_labels = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    print("NumNodes: ", g.number_of_nodes())
    print("NumEdges: ", g.number_of_edges())
    print("NumFeats: ", g.ndata["feat"].shape[1])
    print("NumClasses: ", num_labels)
    print("NumTrainingSamples: ", split_idx["train"].shape[0])
    print("NumValidationSamples: ", split_idx["valid"].shape[0])
    print("NumTestSamples: ", split_idx["test"].shape[0])
    return g, num_labels


def CoraDataset(dev):
    dataset = CoraGraphDataset("./dataset")
    g = dataset[0].to(dev)
    labels = g.ndata["label"]
    num_labels = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    return g, num_labels


def load_dataset(name, dev):
    if name == "cora":
        return CoraDataset(dev)
    else:
        return OgbDataset(name, dev)


def load_pyg_dataset(name, dev):
    import torch_geometric.transforms as T
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.datasets import Planetoid

    if name == "cora":
        path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Planetoid")
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
        data = dataset[0]
    else:
        dataset = PygNodePropPredDataset(name=name, root="/tmp")
        split_idx = dataset.get_idx_split()
        data = dataset[0]  # pyg graph object
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[split_idx["train"]] = True
        data.train_mask = train_mask
        data.y = data.y.view(-1)
    data = data.to(dev)
    return data, dataset


def print_gpu_memory(msg):
    print(msg)
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f"total    : {info.total / 1e9} GB")
    print(f"free     : {info.free / 1e9} GB")
    print(f"used     : {info.used / 1e9} GB")
    a = torch.cuda.memory_allocated(0)
    print(f"Torch allocated: {a / 1e9} GB")
    print(f"Torch max allocated: {torch.cuda.max_memory_allocated(0) / 1e9} GB")


def benchmark(epochs, warmup, model, label, train_mask, *args):
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_fcn = nn.CrossEntropyLoss()

    for epoch in range(epochs + warmup):
        if epoch == warmup:
            torch.cuda.synchronize(0)
            start = time.time()
        model.train()

        # Forward.
        logits = model(*args)

        # Compute loss with nodes in the training set.
        loss = loss_fcn(logits[train_mask], label[train_mask])

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(0)
    end = time.time()
    print(f"Using time: {end - start}, Average time an epoch {(end - start) / epochs}")
    print_gpu_memory("Memory usage during training:")


def benchmark_fw_bw(
    epochs: int,
    warmup: int,
    models: List[callable],
    model_names: Optional[List[str]],
    backward: bool,
    args: Tuple[Any],
):
    from tabulate import tabulate

    if epochs <= 0:
        raise ValueError(f"'epochs' must be a positive integer " f"(got {epochs})")

    if warmup <= 0:
        raise ValueError(f"'warmup' must be a positive integer " f"(got {warmup})")

    ts: List[List[str]] = []
    for model, name in zip(models, model_names):
        t_forward = t_backward = 0
        for i in range(warmup + epochs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            out = model(*args)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if i >= warmup:
                t_forward += time.perf_counter() - t_start

            if backward:
                out_grad = torch.randn_like(out)
                t_start = time.perf_counter()

                out.backward(out_grad)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if i >= warmup:
                    t_backward += time.perf_counter() - t_start

        ts.append([name, f"{t_forward:.4f}s"])
        if backward:
            ts[-1].append(f"{t_backward:.4f}s")
            ts[-1].append(f"{t_forward + t_backward:.4f}s")

    header = ["Name", "Forward"]
    if backward:
        header.extend(["Backward", "Total"])

    print(tabulate(ts, headers=header, tablefmt="psql"))


def benchmark_tensorboard(
    epochs: int,
    models: List[callable],
    model_names: Optional[List[str]],
    args: Tuple[Any],
):
    model_name = models[0].__class__.__name__
    for model, name in zip(models, model_names):
        for _ in range(epochs):
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=5, warmup=1, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f"./log/profile/{model_name}_{name}"
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for _ in range(10):
                    torch.cuda.synchronize()

                    out = model(*args)

                    torch.cuda.synchronize()

                    out_grad = torch.randn_like(out)
                    out.backward(out_grad)
                    prof.step()


def benchmark_profile(epochs, warmup, model, label, train_mask, *args):
    import ScheduleProfiler

    profiler = ScheduleProfiler.ScheduleProfiler()
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_fcn = nn.CrossEntropyLoss()
    for epoch in range(epochs + warmup):
        profiler.start()
        if epoch == warmup:
            torch.cuda.synchronize(0)
            start = time.time()
        model.train()

        # Forward.
        profiler.range_push("forward")
        logits = model(*args)
        profiler.range_pop()

        # Compute loss with nodes in the training set.
        profiler.range_push("loss")
        loss = loss_fcn(logits[train_mask], label[train_mask])
        profiler.range_pop()

        # Backward.
        optimizer.zero_grad()
        profiler.range_push("backward")
        loss.backward()
        profiler.range_pop()
        optimizer.step()

        profiler.stop()
        # print(f"epoch {epoch}, loss: {loss}")
    torch.cuda.synchronize(0)
    end = time.time()
    print(f"Using time: {end - start}, Average time an epoch {(end - start) / epochs}")
    print_gpu_memory("Memory usage during training:")


def benchmark_fn(epochs, warmup, fn):
    for epoch in range(epochs + warmup):
        if epoch == warmup:
            torch.cuda.synchronize(0)
            start = time.time()
        fn()
    torch.cuda.synchronize(0)
    end = time.time()
    print(f"Using time: {end - start}, Average time an epoch {(end - start) / epochs}")
    print_gpu_memory("Memory usage during training:")


def get_func_name(func: Callable) -> str:
    if hasattr(func, "__name__"):
        return func.__name__
    elif hasattr(func, "__class__"):
        return func.__class__.__name__
    raise ValueError("Could not infer name for function '{func}'")


def train(arg, model, label, train_mask, *model_args):
    if arg.compile:
        model = torch.jit.script(model)
        print(model.graph)
        print(model.code)
    if arg.profile:
        benchmark_profile(20, 3, model, label, train_mask, *model_args)
    else:
        benchmark(20, 3, model, label, train_mask, *model_args)


def load_args(parser):
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'ogbn-products', 'ogbn-arxiv').",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    return parser
