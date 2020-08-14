"""
Graph Convolution VAE implementation in pytorch.
The encoder is a graph convolution NN with sparse matrix input of adjacency and edge node attribute matrix.
The decoder a MLP with a flattened normal matrix output.
"""

import time
from torch_rgvae.GCN import GCN
import torch.nn as nn
from torch_rgvae.losses import *
from utils import *
from scipy import sparse


class GCVAE(nn.Module):
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
        n_feat = na + n * ea

        self.encoder = GCN(n, n_feat, h_dim, 2*z_dim).to(torch.double)

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
        bs = A.shape[0]
        # A = sparse.coo_matrix(A)
        # We reshape E to (bs,n,n*d_e) and then concat it with F
        features = np.concatenate((np.reshape(E, (bs, self.n, self.n*self.ea)), F), axis=-1)
        adj = torch.tensor(A)
        # features = sparse.csr_matrix(features)

        # features = self.normalize(features)
        # adj = self.normalize(adj) #+ sp.eye(adj.shape[0]))

        features = torch.Tensor(np.array(features))
        # adj = self.sparse_mx_to_torch_sparse_tensor(adj)

        mean, logstd = torch.split(self.encoder(features, adj), self.z_dim, dim=1)
        return mean, logstd
        
    def decode(self, z):
        logits = self.decoder(z)

        delimit_a = self.n*self.n
        delimit_e = self.n*self.n + self.n*self.n*self.ea

        a, e, f = logits[:,:delimit_a], logits[:,delimit_a:delimit_e], logits[:, delimit_e:]
        A = torch.reshape(a, [-1, self.n, self.n])
        E = torch.reshape(e, [-1, self.n, self.n, self.ea])
        F = self.softmax(torch.reshape(f, [-1, self.n, self.na]))
        return A, E, F
        
    def reparameterize(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        eps = torch.normal(torch.zeros_like(mean), std=1.)
        return eps * torch.exp(logstd) + mean

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

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
        F = torch.reshape(f, [-1, self.n, self.na])
        return A, E, F

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

