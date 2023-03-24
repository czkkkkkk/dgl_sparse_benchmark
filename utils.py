from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset
import torch
import torch.nn as nn
import dgl
import time
from torch.optim import Adam
from pynvml import *
import os.path as osp
import ScheduleProfiler
profiler = ScheduleProfiler.ScheduleProfiler()
# from torch_geometric.datasets import Planetoid
# from ogb.nodeproppred import PygNodePropPredDataset
# import torch_geometric.transforms as T

nvmlInit()

def OgbDataset(graph_name, dev):
    assert graph_name in ('ogbn-products', 'ogbn-arxiv')
    dataset = DglNodePropPredDataset(name=graph_name, root='./dataset')
    g, labels = dataset[0]
    g = dgl.add_self_loop(g).to(dev)
    split_idx = dataset.get_idx_split()
    # get split labels
    g.ndata['label'] = labels.to(dev).view(-1)
    g.ndata['train_mask'] = torch.zeros(g.number_of_nodes(), dtype=torch.bool, device=dev)
    g.ndata['train_mask'][split_idx['train']] = True
    g.ndata['val_mask'] = torch.zeros(g.number_of_nodes(), dtype=torch.bool, device=dev)
    g.ndata['val_mask'][split_idx['valid']] = True
    g.ndata['test_mask'] = torch.zeros(g.number_of_nodes(), dtype=torch.bool, device=dev)
    g.ndata['test_mask'][split_idx['test']] = True
    num_labels = len(
        torch.unique(labels[torch.logical_not(torch.isnan(labels))])
    )
    print('NumNodes: ', g.number_of_nodes())
    print('NumEdges: ', g.number_of_edges())
    print('NumFeats: ', g.ndata['feat'].shape[1])
    print('NumClasses: ', num_labels)
    print('NumTrainingSamples: ', split_idx['train'].shape[0])
    print('NumValidationSamples: ', split_idx['valid'].shape[0])
    print('NumTestSamples: ', split_idx['test'].shape[0])
    return g, num_labels

def CoraDataset(dev):
    dataset = CoraGraphDataset("./dataset")
    g = dataset[0].to(dev)
    labels = g.ndata['label']
    num_labels = len(
        torch.unique(labels[torch.logical_not(torch.isnan(labels))])
    )
    return g, num_labels

def load_dataset(name, dev):
    if name == 'cora':
        return CoraDataset(dev)
    else:
        return OgbDataset(name, dev)

# def load_pyg_dataset(name, dev):
#     if name == 'cora':
#         path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
#         dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
#         data = dataset[0]
#     else:
#         dataset = PygNodePropPredDataset(name=name, root='/tmp') 
#         split_idx = dataset.get_idx_split()
#         train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
#         data = dataset[0] # pyg graph object
#         train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
#         train_mask[split_idx['train']] = True
#         data.train_mask = train_mask
#         data.y = data.y.view(-1)
#     data = data.to(dev)
#     return data, dataset


def print_gpu_memory(msg):
    print(msg)
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total / 1e9} GB')
    print(f'free     : {info.free / 1e9} GB')
    print(f'used     : {info.used / 1e9} GB')
    a = torch.cuda.memory_allocated(0)
    print(f'Torch allocated: {a / 1e9} GB')
    print(f'Torch max allocated: {torch.cuda.max_memory_allocated(0) / 1e9} GB')

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
    print(f'Using time: {end - start}, Average time an epoch {(end - start) / epochs}')
    print_gpu_memory('Memory usage during training:')

def benchmark_profile(epochs, warmup, model, label, train_mask, *args):
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
    print(f'Using time: {end - start}, Average time an epoch {(end - start) / epochs}')
    print_gpu_memory('Memory usage during training:')


def benchmark_fn(epochs, warmup, fn):
    for epoch in range(epochs + warmup):
        if epoch == warmup:
            torch.cuda.synchronize(0)
            start = time.time()
        fn()
    torch.cuda.synchronize(0)
    end = time.time()
    print(f'Using time: {end - start}, Average time an epoch {(end - start) / epochs}')
    print_gpu_memory('Memory usage during training:')