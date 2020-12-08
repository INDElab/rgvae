import torch
import torch.nn as nn
from lp_utils import d
from torch_rgvae.decoders import DistMult



class VEmbed(nn.Module):
    def __init__(self, n_ents: int, n_rels: int, z_dim: int=2):
        super().__init__()
        self.z_dim = z_dim

        # Encoder
        self.s_embed = nn.Embedding(n_ents, 2*z_dim)
        self.r_embed = nn.Embedding(n_rels, 2*z_dim)
        self.o_embed = nn.Embedding(n_ents, 2*z_dim)

        # Decoder
        self.decoder = DistMult(z_dim)
        self.sigmoid = nn.Sigmoid()

    def encode(self, s, r, o):
        """
        """
        bs = s.shape[0]
        # reparametrization
        z_s = self.reparameterize(self.s_embed(s))
        z_r = self.reparameterize(self.r_embed(r))
        z_o = self.reparameterize(self.o_embed(o))
        return (z_s, z_r, z_o)

    def reparameterize(self, mean_logvar):
        """
        Reparametrization trick.
        """
        mean = mean_logvar[:, :, :self.z_dim]
        logvar = mean_logvar[:, :, self.z_dim:]
        eps = torch.normal(torch.zeros_like(mean), std=1.).to(d())
        return eps * torch.exp(logvar * .5) + mean

    def decode(self, z_triple):
        """
        """
        px_z = self.sigmoid(self.decoder(*z_triple))
        return px_z

    def forward(self, s, r, o):
        return self.decode(self.encode(s, r, o))
