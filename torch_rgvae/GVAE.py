"""
Graph-VAE implementation in pytorch.
"""

import time
import torch.nn as nn
from torch_rgvae.losses import *
from utils import *


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

        self.encoder = nn.Sequential(nn.Linear(input_dim, h_dim),
                                    nn.ReLU(),
                                    nn.Dropout(.2),
                                    nn.Linear(h_dim, 2*h_dim),
                                    nn.ReLU(),
                                    nn.Linear(2*h_dim, 2*z_dim))

        self.decoder = nn.Sequential(nn.Linear(z_dim, 2*h_dim),
                                    nn.ReLU(),
                                    nn.Dropout(.2),
                                    nn.Linear(2*h_dim, h_dim),
                                    nn.ReLU(),
                                    nn.Linear(h_dim, input_dim),
                                    nn.Sigmoid())

        self.softmax = nn.Softmax(dim=-1)

        # Need to init?
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight, gain=0.01)
        
    def encode(self, args_in):
        """
        The encoder predicts a mean and logarithm of std of the prior distribution for the decoder.
        Args:
            A: Adjacency matrix of size n*n
            E: Edge attribute matrix of size n*n*ea
            F: Node attribute matrix of size n*na
        """
        (A, E, F) = args_in

        a = torch.reshape(torch.tensor(A * 1.), (-1, self.n*self.n))
        e = torch.reshape(torch.tensor(E * 1.), (-1, self.n*self.n*self.ea))
        f = torch.reshape(torch.tensor(F * 1.), (-1, self.n*self.na))
        x = torch.cat([a, e, f], dim=1)
        mean, logvar = torch.split(self.encoder(x), self.z_dim, dim=1)
        return mean, logvar
        
    def decode(self, z):
        pred = self.decoder(z)
        return self.reconstruct(pred)
        
    def reconstruct(self, pred):
        """
        Reconstructs and returnsthe graph matrices from the flat prediction vector. 
        Args:
            prediciton: the predicted output of the decoder.
        """
        delimit_a = self.n*self.n
        delimit_e = self.n*self.n + self.n*self.n*self.ea

        a, e, f = pred[:,:delimit_a], pred[:,delimit_a:delimit_e], pred[:, delimit_e:]
        A = torch.reshape(a, [-1, self.n, self.n])
        E = torch.reshape(e, [-1, self.n, self.n, self.ea])
        F = self.softmax(torch.reshape(f, [-1, self.n, self.na]))
        return A, E, F

    def reparameterize(self, mean, logvar):
        self.mean = mean
        self.logvar = logvar
        eps = torch.normal(torch.zeros_like(mean), std=1.)
        return eps * torch.exp(logvar * .5) + mean

    def sample(self, n_samples: int=1):
        """
        Sample n times from the model using the target as bernoulli distribution. Return the sampled graph.
        Args:
            n_samples: Number of samples.
        """
        z = torch.randn(n_samples, self.z_dim)
        pred = self.decoder(z)
        b_dist = torch.distributions.Bernoulli(pred)
        samples = b_dist.sample()
        return self.reconstruct(samples)


if __name__ == "__main__":

    my_dtype = torch.float64
    torch.set_default_dtype(my_dtype)

    pass
