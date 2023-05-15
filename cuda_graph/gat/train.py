import argparse
import time

import dgl.nn as dglnn
import pdb

import dgl.sparse as dglsp

import torch
import torch.nn as nn
import torch.nn.functional as F

from cuda_graph.function import graph
from dgl.data import CoraGraphDataset
from torch.optim import Adam
from utils import benchmark_profile


class GATConv(nn.Module):
    def __init__(self, in_size, out_size, num_heads, dropout):
        super().__init__()

        self.out_size = out_size
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.W = nn.Linear(in_size, out_size * num_heads)
        self.a_l = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.a_r = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.W.weight, gain=gain)
        nn.init.xavier_normal_(self.a_l, gain=gain)
        nn.init.xavier_normal_(self.a_r, gain=gain)

    ###########################################################################
    # (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement
    # multihead attention.
    ###########################################################################
    def forward(self, A_hat, Z):
        Z = self.dropout(Z)
        Z = self.W(Z).view(Z.shape[0], self.out_size, self.num_heads)

        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        e_l = (Z * self.a_l).sum(dim=1)
        e_r = (Z * self.a_r).sum(dim=1)
        e = e_l[A_hat.row] + e_r[A_hat.col]

        a = F.leaky_relu(e)
        A_atten = dglsp.val_like(A_hat, a).softmax()
        a_drop = self.dropout(A_atten.val)
        A_atten = dglsp.val_like(A_atten, a_drop)
        return dglsp.bspmm(A_atten, Z)
        # return Z


class GAT(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=8, num_heads=8, dropout=0.6):
        super().__init__()

        self.in_conv = GATConv(
            in_size, hidden_size, num_heads=num_heads, dropout=dropout
        )
        self.out_conv = GATConv(
            hidden_size * num_heads, out_size, num_heads=1, dropout=dropout
        )
        self.stream1 = torch.cuda.Stream()
        

    def forward(self, A_hat, X):
        # Flatten the head and feature dimension.
        current_stream = torch.cuda.current_stream()
        self.stream1.wait_stream(current_stream)
        with torch.cuda.stream(self.stream1):
            Z = F.elu(self.in_conv(A_hat, X)).flatten(1)
            # Average over the head dimension.
            Z = self.out_conv(A_hat, Z).mean(-1)
        current_stream.wait_stream(self.stream1)
        
        return Z


def evaluate(g, pred):
    label = g.ndata["label"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # Compute accuracy on validation/test set.
    val_acc = (pred[val_mask] == label[val_mask]).float().mean()
    test_acc = (pred[test_mask] == label[test_mask]).float().mean()
    return val_acc, test_acc


def train(model, g, A_hat, X):
    label = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_fcn =torch.nn.CrossEntropyLoss()

    for epoch in range(50):
        print("epoch: ", epoch)
        # Forward.
        model.train()
        logits = model(A_hat, X)

        # Compute loss with nodes in training set.
        # loss = loss_fcn(logits[train_mask], label[train_mask])
        loss = loss_fcn(logits, label)
        

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute prediction.
        model.eval()
        logits = model(A_hat, X)
        pred = logits.argmax(dim=1)

        # Evaluate the prediction.
        val_acc, test_acc = evaluate(g, pred)
        print(
            f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f}, test"
            f" acc: {test_acc:.3f}"
        )


if __name__ == "__main__":
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load graph from the existing dataset.
    dataset = CoraGraphDataset()
    g = dataset[0].int().to(dev)
    g = dataset[0].to(dev)

    # Create the sparse adjacency matrix A.
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))

    # Add self-loops.
    I = dglsp.identity(A.shape, device=dev)
    A_hat = A + I

    # Create GAT model
    X = g.ndata["feat"]
    in_size = X.shape[1]
    out_size = dataset.num_classes
    print("in_size", in_size, "out_size", out_size)
    model = GAT(in_size, out_size).to(dev)

    print("Calling graph() pure eager\n")
    stream = torch.cuda.Stream()
    A_hat.requires_grad = False
    
    model = graph(
        model,
        sample_args=(A_hat,X.clone(),),
        graph_stream=stream,
        warmup_iters=0,
        warmup_only=True,
        overwrite_fn="eager_forward",
    )
    model = graph(
        model,
        sample_args=(A_hat,X.clone(),),
        graph_stream=stream,
        warmup_only=False,
        overwrite_fn="graph_forward",
    )
    # pdb.set_trace()

    class GraphedWrapper(torch.nn.Module):
        def __init__(self, model_segment):
            super().__init__()
            self.model_segment = model_segment

        def forward(self, g, x):
            if self.training:
                return self.model_segment.graph_forward(g,x)
            else:
                return self.model_segment.eager_forward(g,x)

    model = GraphedWrapper(model)
    # Kick off training.
    
    train(model, g, A_hat, X)
    # benchmark_profile(20, 5, model, g.ndata["label"], g.ndata["train_mask"], A_hat, X)
