"""
Graph-VAE implementation in pytorch.
Also parent class for all other VAE models.
"""
import time
import torch.nn as nn
from torch_rgvae.encoders import *
from torch_rgvae.decoders import *
from torch_rgvae.losses import *
from utils import *
from lp_utils import d


class GVAE(nn.Module):
    def __init__(self, n: int, n_r: int, n_e: int, dataset_name: str, h_dim: int=1024, z_dim: int=2, beta: float=1., softmax_E: bool=True):
        """
        Graph Variational Auto Encoder
        :param n : Number of nodes
        :param n_e : Number of node attributes
        :param n_r : Number of edge attributes
        :param dataset_name : name of the dataset which the model will train on.
        :param h_dim : Hidden dimension
        :param z_dim : latent dimension
        :param beta: for beta < 1, makes the model is a beta-VAE
        :param softmax_E : use softmax for edge attributes
        """
        super().__init__()
        self.name = 'GVAE'
        self.n = n
        self.n_e = n_e
        self.n_r = n_r
        input_dim = n*n + n*n_e + n*n*n_r
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.beta = torch.tensor(beta)
        self.softmax_E = softmax_E
        self.dataset_name = dataset_name

        self.encoder = MLP(input_dim, h_dim, z_dim)

        self.decoder = RMLP(input_dim, h_dim, z_dim)

        self.softmax = nn.Softmax(dim=-1)

        # Need to init?
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
        
    def encode(self, args_in):
        """
        The encoder predicts a mean and logarithm of std of the prior distribution for the decoder.
        Args:
            A: Adjacency matrix of size n*n
            E: Edge attribute matrix of size n*n*n_r
            F: Node attribute matrix of size n*n_e
        """
        (A, E, F) = args_in
        self.edge_count = torch.norm(A[0], p=1)

        a = torch.reshape(A, (-1, self.n*self.n))
        e = torch.reshape(E, (-1, self.n*self.n*self.n_r))
        f = torch.reshape(F, (-1, self.n*self.n_e))
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
        delimit_e = self.n*self.n + self.n*self.n*self.n_r

        a, e, f = pred[:,:delimit_a], pred[:,delimit_a:delimit_e], pred[:, delimit_e:]
        A = torch.reshape(a, [-1, self.n, self.n])
        E = torch.reshape(e, [-1, self.n, self.n, self.n_r])
        if self.softmax_E:
            E = self.softmax(E)
        F = self.softmax(torch.reshape(f, [-1, self.n, self.n_e]))
        return A, E, F

    def reparameterize(self, mean, logvar):
        """
        Reparametrization trick.
        """
        self.mean = mean
        self.logvar = logvar
        eps = torch.normal(torch.zeros_like(mean), std=1.).to(d())
        return eps * torch.exp(logvar * .5) + mean

    def forward(self, target):
        """
        Forward pass of the VAE.
        :param target: batch of graphs in spare form.
        :return : Prediction of the model.
        """
        mean, logvar = self.encode(target)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)

    def reconstruction_loss(self, target, prediction):
        loss, x_permute = mpgm_loss(target, prediction)
        self.x_permute = x_permute
        return loss
    
    def regularization_loss(self, mean, logvar):
        return kl_divergence(mean, logvar)

    def elbo(self,target):
        """
        Loss function of the VAE.
        :param target: The target Graph.
        :return : the ELBO loss
        """
        mean, logvar = self.encode(target)
        z = self.reparameterize(mean, logvar)
        prediction = self.decode(z)
        return self.beta * self.regularization_loss(mean, logvar) - self.reconstruction_loss(target, prediction)

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
