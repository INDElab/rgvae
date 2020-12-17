"""
Decoders for the GraphVAE.
"""
import torch
import torch.nn as nn
from abc import abstractmethod


class RMLP(nn.Module):
    """
    Simple reverse multi layer perceptron.
    """
    def __init__(self, input_dim, h_dim, z_dim):
        super().__init__()
        self.rmlp = nn.Sequential(nn.Linear(z_dim, 2*h_dim),
                                    nn.ReLU(),
                                    nn.Dropout(.2),
                                    nn.Linear(2*h_dim, h_dim),
                                    nn.ReLU(),
                                    nn.Linear(h_dim, input_dim))

    def forward(self, x):
        return self.rmlp(x)


class Decoder(nn.Module):
    """
    Source:https://github.com/pbloem/embed/lpmodels.py
    """

    def __init__(self, e):
        """
        :param e: latent dimension
        """
        super().__init__()

        self.e = e

    def s_dim(self):
        return self.e

    def p_dim(self):
        return self.e

    def o_dim(self):
        return self.e

    @abstractmethod
    def forward(self, triples, corruptions, corr_index, entities, relations, forward):
        pass


class DistMult(Decoder):
    """
    Source:https://github.com/pbloem/embed/lpmodels.py
    """
    def __init__(self, e):
        super().__init__(e)
        self.e = e      # Latent dimensions

    def forward(self, s, p, o):
        """
        Implements the distmult score function.
        :param s: subject embedded
        :param r: relation embedded
        :param o: object embedded
        """
        # TODO fix the si pi oi thing 

        if len(s.size()) == len(p.size()) == len(o.size()): # optimizations for common broadcasting
            if p.size(-2) == 1 and o.size(-2) == 1:
                singles = p * o # ignoring batch dimensions, this is a single vector
                return torch.matmul(s, singles.transpose(-1, -2)).squeeze(-1)

            if s.size(-2) == 1 and o.size(-2) == 1:
                singles = s * o
                return torch.matmul(p, singles.transpose(-1, -2)).squeeze(-1)

            if s.size(-2) == 1 and p.size(-2) == 1:
                singles = s * p
                return torch.matmul(o, singles.transpose(-1, -2)).squeeze(-1)

        return (s * p * o).sum(dim=-1)
