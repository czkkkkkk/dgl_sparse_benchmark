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