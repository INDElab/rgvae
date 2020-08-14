"""
Thomas Kipf's code <3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_rgvae.layers.GCN_layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, n, nfeat, nhid, nclass, dropout= .2):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.dense = nn.Linear(n*nclass, nclass)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.dense(x)
        return x
        