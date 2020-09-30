import pytest
import numpy as np
import torch
from graph_matching.MPGM import MPGM

# This sets the default torch dtype. Double-power
my_dtype = torch.float64
torch.set_default_dtype(my_dtype)
mpgm = MPGM()

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
F = torch.randint(2, (batch_size,n,d_n)))
A_hat = torch.randn((batch_size,k,k))
E_hat = torch.randn((batch_size,k,k,d_e))
F_hat = torch.randn((batch_size,k,d_n))

S = torch.randn((batch_size,n,n,k,k))
class TestMPGM():
    """
    Test class for the max-pooling graph matching algorithm.
    """
    def test_affinity(self):
        # First assert shape
        affintiy_matrix_batch = mpgm.affinity(A, A_hat, E, E_hat, F, F_hat)
        assert affintiy_matrix_batch.shape == torch.Size([batch_size,n,n,k,k])

        # TODO: Assert no inf or nan

        # Second assert the correct values using the batch alg vs the 
        affintiy_matrix_sci_0 = mpgm.affinity_loop(A[0], A_hat[0], E[0], E_hat[0], F[0], F_hat[0])
        affintiy_matrix_sci_1 = mpgm.affinity_loop(A[1], A_hat[1], E[1], E_hat[1], F[1], F_hat[1])        
        affintiy_matrix_batch_0 = affintiy_matrix_batch[0].squeeze().numpy()
        affintiy_matrix_batch_1 = affintiy_matrix_batch[1].squeeze().numpy()
        assert affintiy_matrix_batch_0.all() == affintiy_matrix_sci.all()
        assert affintiy_matrix_batch_1.all() == affintiy_matrix_sci_1.all()

    def test_maxpool(self):
        Xs_batch = mpgm.max_pool(S)
        assert Xs_batch.shape == torch.Size(batch_size,n,k)

        Xs_sci_0 = mpgm.max_pool_loop(S[0].squeeze().numpy())
        Xs_sci_1 = mpgm.max_pool_loop(S[1].squeeze().numpy())
        assert Xs_batch[0].squeeze().numpy() == Xs_sci_0
        assert Xs_batch[1].squeeze().numpy() == Xs_sci_1

    def test_hungarian(self):
        pass



test = TestMPGM()
test.test_affinity()
test.test_maxpool()
test.test_hungarian()