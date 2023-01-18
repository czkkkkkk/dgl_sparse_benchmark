"""
[SIGN: Scalable Inception Graph Neural Networks]
(https://arxiv.org/abs/2004.11198)

This example shows a simplified version of SIGN: a precomputed 2-hops diffusion
operator on top of symmetrically normalized adjacency matrix A_hat.
"""

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils import load_dataset, benchmark
import argparse

################################################################################
# (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement the feature
# diffusion in SIGN laconically.
################################################################################


def sign_diffusion(A, X, r):
    # Perform the r-hop diffusion operation.
    X_sign = [X]
    print(type(X))
    for _ in range(r):
        X = A @ X
        X_sign.append(X)
    X_sign = torch.cat(X_sign, dim=1)
    return X_sign


class SIGN(nn.Module):
    def __init__(self, in_size, out_size, r, hidden_size=256):
        super().__init__()
        # Note that theta and omega refer to the learnable matrices in the
        # original paper correspondingly. The variable r refers to subscript to
        # theta.
        self.theta = nn.Linear(in_size * (r + 1), hidden_size * (r + 1))
        self.omega = nn.Linear(hidden_size * (r + 1), out_size)

    def forward(self, X_sign: torch.Tensor):
        results = self.theta(X_sign)
        Z = F.relu(results)
        return self.omega(Z)


def evaluate(g, pred):
    label = g.ndata["label"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # Compute accuracy on validation/test set.
    val_acc = (pred[val_mask] == label[val_mask]).float().mean()
    test_acc = (pred[test_mask] == label[test_mask]).float().mean()
    return val_acc, test_acc


def train(model, g, X_sign):
    label = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    optimizer = Adam(model.parameters(), lr=3e-3)

    for epoch in range(10):
        # Switch the model to training mode.
        model.train()

        # Forward.
        logits = model(X_sign)

        # Compute loss with nodes in training set.
        loss = F.cross_entropy(logits[train_mask], label[train_mask])

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Switch the model to evaluating mode.
        model.eval()

        # Compute prediction.
        logits = model(X_sign)
        pred = logits.argmax(1)

        # Evaluate the prediction.
        val_acc, test_acc = evaluate(g, pred)
        print(
            f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f}, test"
            f" acc: {test_acc:.3f}"
        )


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

    # Create the sparse adjacency matrix A (note that W was used as the notation
    # for adjacency matrix in the original paper).
    src, dst = g.edges()
    N = g.num_nodes()
    A = dglsp.from_coo(dst, src, shape=(N, N))
    label = g.ndata["label"]
    train_mask = g.ndata["train_mask"]

    # Calculate the symmetrically normalized adjacency matrix.
    I = dglsp.identity(A.shape, device=dev)
    A_hat = A + I
    D_hat = dglsp.diag(A_hat.sum(dim=1)) ** -0.5
    A_hat = D_hat @ A_hat @ D_hat

    # 2-hop diffusion.
    r = 2
    X = g.ndata["feat"]
    X_sign = sign_diffusion(A_hat, X, r)

    # Create SIGN model.
    in_size = X.shape[1]
    out_size = num_classes
    model = SIGN(in_size, out_size, r).to(dev)

    # Kick off training.
    train(model, g, X_sign)
    benchmark(200, 3, model, label, train_mask, X_sign)

    model_script = torch.jit.script(model)
    print(model_script.graph)
    print(model_script.code)

    benchmark(200, 3, model_script, label, train_mask, X_sign)
