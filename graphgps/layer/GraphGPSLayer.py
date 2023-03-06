import torch
import torch.nn as nn
import torch.nn.functional as F
from SparseMHA import SparseMHA

class GraphGPSLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=80, num_heads=8, dropout=0, batch_norm=True):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.norm_local = nn.BatchNorm1d(hidden_size)
            self.norm_attn = nn.BatchNorm1d(hidden_size)
            self.norm_out = nn.BatchNorm1d(hidden_size)
            
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)
        
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
        
        
    def forward(self, A, h):
        h_out_list = []
        h_in = h
        
        # Local MPNN
        h_local = self.local_model()
        h_local = self.dropout_local(h_local)
        h_local = h_in + h_local
        if self.batch_norm:
            h_local = self.norm_local(h_local)
        
        # Multi-head attention
        h_attn = self.MHA(A, h)
        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in + h_attn
        if self.batch_norm:
            h_attn = self.norm_attn(h_attn)

        # Combine the local and global outputs
        h = h_local + h_attn
        h = h + self.FFN2(F.relu(self.FFN1(h)))
        if self.batch_norm:
            h = self.norm_out(h)
        return h