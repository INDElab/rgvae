"""
Graph-VAE implementation in pytorch.
Encoder: RGCNN
Decoder: MLP
"""

import torch.nn as nn
from torch_rgvae.RGVAE import *
from torch_rgvae.GVAE import GVAE


class TorchRGVAE(GVAE):
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
        super().__init__(n, ea, na, h_dim, z_dim)
        self.name = 'RGVAE'
        self.n = n
        self.na = na
        self.ea = ea
        input_dim = n*n + n*na + n*n*ea
        self.input_dim = input_dim
        self.z_dim = z_dim

        nfeat = na + ea * n         # We concat the edge attribute matrix with the node attribute matrix. Lets see what happens.
        self.encoder = NodeClassifier(triples=None,
                                        nnodes=n,
                                        nrel=ea,
                                        nfeat=None,
                                        nhid=h_dim,
                                        nlayers=2,
                                        nclass=None,
                                        edge_dropout=None,
                                        decomposition=None,
                                        nemb=None)

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

    
    def encode(self, args_in):
        """
        The encoder predicts a mean and logarithm of std of the prior distribution for the decoder.
        A simple MLP decoder with a RGCN encoder.
        """
        x = self.encoder()
        print(x)
        # TODO: Split x!!!)
        return mean, logstd
        

if __name__ == "__main__":
    pass
