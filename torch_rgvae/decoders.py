"""
Decoders for the GraphVAE.
"""
import torch
import torch.nn as nn


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
                                    nn.Linear(h_dim, input_dim),
                                    nn.Sigmoid())

    def forward(self, x):
        return self.rmlp(x)