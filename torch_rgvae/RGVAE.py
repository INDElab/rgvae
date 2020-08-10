"""
Graph-VAE implementation in pytorch.
Encoder: RGCNN
Decoder: MLP
"""

import torch.nn as nn
from torch_rgvae.RGCN_models import *


class TorchRGVAE(nn.Module):
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
        # Split x!!!)
        return mean, logstd
        
    def decode(self, z):
        logits = self.decoder(z)

        delimit_a = self.n*self.n
        delimit_e = self.n*self.n + self.n*self.n*self.ea

        a, e, f = logits[:,:delimit_a], logits[:,delimit_a:delimit_e], logits[:, delimit_e:]
        A = torch.reshape(a, [-1, self.n, self.n])
        E = torch.reshape(e, [-1, self.n, self.n, self.ea])
        F = torch.reshape(f, [-1, self.n, self.na])
        return A, E, F
        
    def reparameterize(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        eps = torch.normal(torch.zeros_like(mean), std=1.)
        return eps * torch.exp(logstd) + mean

if __name__ == "__main__":
    pass
