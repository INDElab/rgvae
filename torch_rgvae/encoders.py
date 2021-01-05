"""
Collection of encoders to use in the GraphVAE.
"""
from torch_rgvae.layers.RGC_layers import RelationalGraphConvolution
import torch.nn.functional as F
from torch import nn
import torch
from torch_rgvae.layers.GCN_layers import GraphConvolution


class MLP(nn.Module):
    """
    Simple multi layer perceptron
    """
    def __init__(self, input_dim, h_dim, z_dim):
        super().__init__()

        self.mlp = nn.Sequential(nn.Linear(input_dim, 2*h_dim),
                                            nn.ReLU(),
                                            nn.Dropout(.2),
                                            nn.Linear(2*h_dim, h_dim),
                                            nn.ReLU(),
                                            nn.Linear(h_dim, z_dim))

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    """
    Graph convolution net.
    Source: tkipf gcnn
    """
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
        

class NodeClassifier(nn.Module):
    """ Node classification with R-GCN message passing """
    def __init__(self,
                 triples=None,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 nhid=16,
                 nlayers=2,
                 nclass=None,
                 edge_dropout=None,
                 decomposition=None,
                 nemb=None):
        super(NodeClassifier, self).__init__()

        self.nlayers = nlayers

        assert (triples is not None or nnodes is not None or nrel is not None or nclass is not None), \
            "The following must be specified: triples, number of nodes, number of relations and number of classes!"
        assert 0 < nlayers < 3, "Only supports the following number of RGCN layers: 1 and 2."

        if nlayers == 1:
            nhid = nclass

        if nlayers == 2:
            assert nhid is not None, "Number of hidden layers not specified!"

        triples = torch.tensor(triples, dtype=torch.long)
        with torch.no_grad():
            self.register_buffer('triples', triples)
            # Add inverse relations and self-loops to triples
            self.register_buffer('triples_plus', add_inverse_and_self(triples, nnodes, nrel))

        self.rgc1 = RelationalGraphConvolution(
            triples=self.triples_plus,
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,
            in_features=nfeat,
            out_features=nhid,
            edge_dropout=edge_dropout,
            decomposition=decomposition,
            vertical_stacking=False
        )
        if nlayers == 2:
            self.rgc2 = RelationalGraphConvolution(
                triples=self.triples_plus,
                num_nodes=nnodes,
                num_relations=nrel * 2 + 1,
                in_features=nhid,
                out_features=nclass,
                edge_dropout=edge_dropout,
                decomposition=decomposition,
                vertical_stacking=True
            )

    def forward(self):
        """ Embed relational graph and then compute class probabilities """
        x = self.rgc1()

        if self.nlayers == 2:
            x = F.relu(x)
            x = self.rgc2(features=x)

        return x

