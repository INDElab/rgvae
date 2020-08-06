"""
Graph-VAE implementation in pytorch.
"""

import time
import torch.nn as nn
import torch
from torch.nn import functional as F
from models.losses import *
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
        mean, logstd = torch.split(self.encoder(x), self.z_dim, dim=1)
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

    my_dtype = torch.float64
    torch.set_default_dtype(my_dtype)

    n = 5
    d_e = 3
    d_n = 2
    np.random.seed(seed=11)
    epochs = 111
    batch_size = 64

    train_set = mk_random_graph_ds(n, d_e, d_n, 400, batch_size=batch_size)
    test_set = mk_random_graph_ds(n, d_e, d_n, 100, batch_size=batch_size)

    model = TorchGVAE(n, d_e, d_n)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        start_time = time.time()

        for target in train_set:
            model.train()
            mean, logstd = model.encode(target)
            z = model.reparameterize(mean, logstd)
            prediction = model.decode(z)

            log_pz = log_normal_pdf(z, torch.zeros_like(z), torch.zeros_like(z))
            log_qz_x = log_normal_pdf(z, mean, 2*logstd)
            log_px = mpgm_loss(target, prediction)
            loss = - torch.mean(log_px + log_pz + log_qz_x)
            print(loss)
            loss.backward()
            optimizer.step()
            end_time = time.time()

        # Evaluate
        mean_loss = []
        with torch.no_grad():
            model.eval()
            for test_x in test_set:
                mean, logstd = model.encode(target)
                z = model.reparameterize(mean, logstd)
                prediction = model.decode(z)
                log_pz = log_normal_pdf(z, torch.zeros_like(z), torch.zeros_like(z))
                log_qz_x = log_normal_pdf(z, mean, 2*logstd)
                log_px = mpgm_loss(target, prediction)
                loss = - torch.mean(log_px + log_pz + log_qz_x)
                mean_loss.append(loss)
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, np.mean(mean_loss), end_time - start_time))

