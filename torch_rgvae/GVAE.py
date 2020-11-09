"""
Graph-VAE implementation in pytorch.
Also parent class for all other VAE models.
"""

import time
import torch.nn as nn
from torch_rgvae.losses import *
from torch_rgvae.encoders import *
from torch_rgvae.decoders import *
from utils import *


class GVAE(nn.Module):
    def __init__(self, n: int, ea: int, na: int, h_dim: int=512, z_dim: int=2, softmax_E: bool=True):
        """
        Graph Variational Auto Encoder
        :param n : Number of nodes
        :param na : Number of node attributes
        :param ea : Number of edge attributes
        :param h_dim : Hidden dimension
        :param z_dim : latent dimension
        :param softmax_E : use softmax for edge attributes
        """
        super().__init__()
        self.name = 'GVAE'
        self.n = n
        self.na = na
        self.ea = ea
        input_dim = n*n + n*na + n*n*ea
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.softmax_E = softmax_E

        self.encoder = MLP(input_dim, h_dim, z_dim)

        self.decoder = RMLP(input_dim, h_dim, z_dim)

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
        (a, e, f) = args_in
        self.edge_count = torch.norm(a[0], p=1)

        x = torch.cat([a, e, f], dim=1)
        mean, logvar = torch.split(self.encoder(x), self.z_dim, dim=1)
        return mean, logvar
        
    def decode(self, z):
        self.z = z
        pred = self.decoder(z)
        return self.reconstruct(pred)
        
    def reconstruct(self, pred):
        """
        Reconstructs and returns the graph matrices from the flat prediction vector. 
        Args:
            prediction: the predicted output of the decoder.
        """
        delimit_a = self.n*self.n
        delimit_e = self.n*self.n + self.n*self.n*self.ea

        a, e, f = pred[:,:delimit_a], pred[:,delimit_a:delimit_e], pred[:, delimit_e:]
        A = torch.reshape(a, [-1, self.n, self.n])
        E = torch.reshape(e, [-1, self.n, self.n, self.ea])
        if self.softmax_E:
            E = self.softmax(E)
        F = self.softmax(torch.reshape(f, [-1, self.n, self.na]))
        return A, E, F

    def reparameterize(self, mean, logvar):
        self.mean = mean
        self.logvar = logvar
        eps = torch.normal(torch.zeros_like(mean), std=1.)
        return eps * torch.exp(logvar * .5) + mean

    def forward(self, args_in):
        """
        Forward pass of the VAE.
        :param args_in: batch of graphs in spare form.
        :return : Prediction of the model.
        """
        mean, logvar = self.encode(args_in)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)

    def sample(self, z, n_samples: int=1):
        """
        Sample n times from the model using the target as bernoulli distribution. Return the sampled graph.
        Args:
            z:  Decoder input signal, shape (batch_size, z_dim)
        """
        assert z.shape[-1] == self.z_dim
        a, e, f = self.reconstruct(self.decoder(z))
        a_dist = torch.distributions.Bernoulli(a)
        a_sample = a_dist.sample()
        if self.softmax_E:
            # in this case e will be dense
            e_dist = torch.distributions.Categorical(e)
        else:
            e_dist = torch.distributions.Bernoulli(e)
        e_dense = e_dist.sample()
        f_dist = torch.distributions.Categorical(f)
        f_dense = f_dist.sample()


        return (a_sample, e_dense, f_dense)

    def sanity_check(self):
        """
        Function to monitor the sanity logic of the prediction.
        Sanity 1: Model should predict graphs with the same amount of nodes as the target graph.
        Sanity 2: Model should predict graphs with the same amount of edges as the target graph.
        Sanity 3: Model should only predict edge attributes were it also predicts edges.
        Args:
            sample: binary prediction sample.
            n: number of nodes in target graph.
            e: number of edges in target graph.
        Returns:
            The 3 sanities in percentage.
        """
        A, E, F = self.sample(self.z)
        n, e = (self.n, self.edge_count)

        # Sanity 1
        A_check = A.detach().clone().cpu().numpy()
        A_check = A_check[~np.all(A_check == 0, axis=1)]
        A_check = np.delete(A_check, np.where(~A_check.any(axis=1))[0], axis=0)
        k = A_check.shape[np.argmax(A_check.shape)] * 1.
        if k <= n:
            sanity_1 = k/n
        else:
            sanity_1 = 1 - (k-n)/n
        
        # Sanity 2
        e_check = np.sum(A_check)
        if e_check <= e:
            sanity_2 = e_check/e
        else:
            sanity_2 = 1 - (e_check-e)/e

        return sanity_1 * 100, sanity_2 * 100


if __name__ == "__main__":

    my_dtype = torch.float64
    torch.set_default_dtype(my_dtype)

    pass
