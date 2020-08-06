"""
Main script.
For now it executes the training and evaluation of the RGVAE with radom data.
Change the parameters in-script. Parsing is yet to come.
"""
import torch
import numpy as np
from train import *


# This sets the default torch dtype. Double-power
my_dtype = torch.float64
torch.set_default_dtype(my_dtype)

# Parameters. Arg parsing on its way.
n = 5
e = 10      # All random graphs shall have 5 nodes and 10 edges
d_e = 3
d_n = 3
batch_size = 64
params = [n, e, d_e, d_n, batch_size]

seed = 11
np.random.seed(seed=seed)
torch.manual_seed(seed)
epochs = 111
lr = 1e-5

train_eval_GVAE(params, epochs, lr)
