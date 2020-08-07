"""
Graph-VAE implementation in pytorch.
Encoder: RGCNN
Decoder: MLP
"""

import torch.nn as nn
from GCN import *


class TorchGVAE(nn.Module):
    def __init__(self, n: int, ea: int, na: int, h_dim: int=512, z_dim: int=2):
        """
        Graph Variational Auto Encoder
        Args:
            n : Number of nodes
            na : Number of node attributes
            ea : Number of edge attributes
            h_dim : Hidden dimension
            z_dim : latent dimension
        """
        super().__init__()
        self.n = n
        self.na = na
        self.ea = ea
        input_dim = n*n + n*na + n*n*ea
        self.input_dim = input_dim
        self.z_dim = z_dim

        nfeat = na + ea * n         # We concat the edge attribute matrix with the node attribute matrix. Lets see what happens.
        self.encoder = GCN(nfeat, h_dim, z_dim)
        self.decoder = nn.Sequential(nn.Linear(z_dim, 2*h_dim),
                                    nn.ReLU(),
                                    nn.Dropout(.2),
                                    nn.Linear(2*h_dim, h_dim),
                                    nn.ReLU(),
                                    nn.Linear(h_dim, input_dim),
                                    nn.Sigmoid())

        # Need to init?
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight, gain=0.01)

        pass
    