"""
Graph-VAE implementation in pytorch.
Encoder: RGCNN
Decoder: MLP
"""

import torch
import torch.nn as nn
from torch_rgvae.encoders import NodeClassifier
from torch_rgvae.GVAE import GVAE
from torch_rgvae.decoders import RMLP, sRMLP


class TorchRGVAE(GVAE):
    def __init__(self, args, n_r: int, n_e: int, data, dataset_name: str):
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
        super().__init__(args, n_r, n_e, dataset_name)
        self.name = 'RGVAE'
        self.n = n = args['n']
        self.n_e = n_e
        self.n_r = n_r
        self.z_dim = z_dim = args['z_dim'] if 'z_dim' in args else 2
        self.h_dim = h_dim = args['h_dim'] if 'h_dim' in args else 1024
        beta = args['beta'] if 'beta' in args else 1.
        self.delta = args['delta'] if 'delta' in args else 0.
        self.beta = torch.tensor(beta)
        self.softmax_E = args['softmax_E'] if 'softmax_E' in args else True
        self.perm_inv = args['perm_inv'] if 'perm_inv' in args else True
        self.adj_argmax = args['adj_argmax'] if 'adj_argmax' in args else True
        self.clip_grad = args['clip_grad'] if 'clip_grad' in args else True
        self.dataset_name = dataset_name
        self.model_params = args
        self.k = k = n                          # assumption n=k

        self.encoder = NodeClassifier(triples=data,
                                        nnodes=n_e,
                                        nrel=n_r,
                                        nfeat=None,
                                        nhid=h_dim,
                                        nlayers=1,
                                        nclass=2*z_dim,
                                        edge_dropout=None,
                                        decomposition=None,)

        self.decoder = sRMLP(k, 1+n_r, h_dim, self.z_dim)


        # Need to init?
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight, gain=0.01)

    
    def encode(self, triples):
        """
        The encoder predicts a mean and logarithm of std of the prior distribution for the decoder.
        A simple MLP decoder with a RGCN encoder.
        """
        assert len(triples) == self.n
        x = self.encoder(triples)
        mean, logvar = torch.split(x, self.z_dim, dim=1)
        return mean, logvar
        
    def decode(self, z):
        # new decoder: 15000,2 --> k,15000,15000*400
        self.z = z
        pred = self.decoder(z).view(self.k, -1)
        return self.reconstruct(pred)
        
    def reconstruct(self, pred):
        """
        Reconstructs and returns the graph matrices from the flat prediction vector. 
        Args:
            prediction: the predicted output of the decoder.
        """
        idx2, idx1 = torch.split(pred, self.n_e*self.n_r, dim=1)
        idx1 = torch.argmax(idx1, dim=-1)
        idx2 = torch.argmax(idx2.view(self.k, self.n_e, -1).sum(-1), dim=-1)
        idxr = torch.floor(idx2/self.n_e)
        idx2 = idx2 - idxr*self.n_e
        
        return torch.cat([idx1.unsqueeze(-1),idxr.unsqueeze(-1),idx2.unsqueeze(-1)], dim=-1)
        

if __name__ == "__main__":
    pass
