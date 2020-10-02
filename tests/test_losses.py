import numpy as np
import torch
from torch_rgvae.losses import graph_BCEloss, mpgm_loss, kl_divergence

# This sets the default torch dtype. Double-power
my_dtype = torch.float64
torch.set_default_dtype(my_dtype)

# Let's define some dimensions :)
n = 3
k = 3
d_e = 100
d_n = 50

batch_size = 2
seed = 11
# Generation of random test graphs. The target graph is discrete and the reproduced graph probabilistic.
torch.manual_seed(seed)
np.random.seed(seed=seed)
A = torch.randint(2, (batch_size,n,n))
E = torch.randint(2, (batch_size,n,n,d_e))
F = torch.randint(2, (batch_size,n,d_n))
A_hat = torch.rand((batch_size,k,k))
E_hat = torch.rand((batch_size,k,k,d_e))
F_hat = torch.rand((batch_size,k,d_n))


def test_graph_BCEloss():
    loss_equal = graph_BCEloss([A,E,F],[A.to(my_dtype),E.to(my_dtype),F.to(my_dtype)])
    loss = graph_BCEloss([A,E,F],[A_hat,E_hat,F_hat])
    assert loss_equal.numpy().all() == 0.
    assert loss.numpy().all() != 0.

def test_mgpm_loss():
    pass
test_graph_BCEloss()
