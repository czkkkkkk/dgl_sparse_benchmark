import torch
from torch import nn
import torch.functional as F
import numpy as np
import functools
import math, copy
import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from tqdm import tqdm


self_attn = torch.nn.MultiheadAttention(64, 4, dropout=0, batch_first=True).to(
    "cuda:0"
)
torch.nn.init.constant_(self_attn.in_proj_weight, 1)
torch.nn.init.constant_(self_attn.in_proj_bias, 0)
torch.nn.init.constant_(self_attn.out_proj.weight, 1)
torch.nn.init.constant_(self_attn.out_proj.bias, 0)



def _sa_block(x, attn_mask, key_padding_mask):
    """Self-attention block."""
    x = self_attn(
        x,
        x,
        x,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=False,
    )[0]
    return x

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = nn.Softmax(dim=-1)(scores)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList()
        for _ in range(4):
            self.linears.append(nn.Linear(d_model, d_model))
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(4):
            torch.nn.init.constant_(self.linears[i].weight, 1)
            torch.nn.init.constant_(self.linears[i].bias, 0)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        print(
            "Before transform query: " + str(query.size())
        )  # (batch_size, seq_length, d_model)
        query, key, value = [
            linear(x) for linear, x in zip(self.linears, (query, key, value))
        ]  # (batch_size, seq_length, d_model), use first 3 self.linears
        query, key, value = [
            x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for x in (query, key, value)
        ]  # (batch_size, h, seq_length, d_k)

        print("After transform query: " + str(query.size()))

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        print("After attention: " + str(x.size()))

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).reshape(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)

class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, hidden_size=80, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.constant_(self.q_proj.weight, 1)
        torch.nn.init.constant_(self.q_proj.bias, 0)
        
        torch.nn.init.constant_(self.k_proj.weight, 1)
        torch.nn.init.constant_(self.k_proj.bias, 0)
        
        torch.nn.init.constant_(self.v_proj.weight, 1)
        torch.nn.init.constant_(self.v_proj.bias, 0)
        
        torch.nn.init.constant_(self.out_proj.weight, 1)
        torch.nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, A, h):
        N = len(h)
        # [N, dh, nh]
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        # [N, dh, nh]
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        # [N, dh, nh]
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)

        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        if A is not None:
            attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        else:
            attn = dglsp.bspmm(q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        # Sparse softmax by default applies on the last sparse dimension.
        attn = attn.softmax()  # (sparse) [N, N, nh]
        out = dglsp.bspmm(attn, v)  # [N, dh, nh]

        return self.out_proj(out.reshape(N, -1))

def main():
    # load data
    h_dense = torch.load("./log/h_dense.pt").to("cuda:0")
    mask = torch.load("./log/mask.pt").to("cuda:0")
    h_attn_origin = torch.load("./log/h_attn.pt").to("cuda:0")
    h = torch.load("./log/h.pt").to("cuda:0")
    batch = torch.load("./log/batch.pt").to("cuda:0")

    # origin implementation
    h_attn_dense = _sa_block(h_dense, None, ~mask)
    h_attn = h_attn_dense[mask]

    print("running success")
    print(torch.isclose(h_attn, h_attn_origin))
    print(
        f"""h_attn {h_attn.shape}, h_dense {h_dense.shape}
        mask {mask.shape}, h_attn_dense {h_attn_dense.shape}
        {h.shape}, batch{batch.shape}"""
    )

    ## naive implementation
    attn_block = MultiHeadAttention(4, 64).to("cuda:0")
    h_attn_dense_naive = attn_block(h_dense, h_dense, h_dense)
    h_attn_naive = h_attn_dense_naive[mask]
    print("running success")
    result = torch.isclose(h_attn_naive, h_attn_origin)
    for _ in range(2):
        result = functools.reduce(lambda x,y: x|y, result)
    print(result.item())
    
    # DGL implementation
    attn_sparse = SparseMHA(64, 4).to("cuda:0")
    h_attn_sparse = attn_sparse(None, h)
    
if __name__ == "__main__":
    main()
