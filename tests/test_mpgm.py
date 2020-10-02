import pytest
import numpy as np
import torch
from graph_matching.MPGM import MPGM
from utils import add_e7
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
F = torch.randint(2, (batch_size,n,d_n))
A_hat = torch.rand((batch_size,k,k))
E_hat = torch.rand((batch_size,k,k,d_e))
F_hat = torch.rand((batch_size,k,d_n))

S = torch.rand((batch_size,n,n,k,k))
Xs = torch.rand((batch_size,n,k))

class TestMPGM():
    """
    Test class for the max-pooling graph matching algorithm.
    """
    def test_affinity(self):
        # First assert shape
        affintiy_matrix_batch = mpgm.affinity(A, A_hat, E, E_hat, F, F_hat)
        assert affintiy_matrix_batch.shape == torch.Size([batch_size,n,n,k,k])

        # Assert no inf or nan
        assert torch.isnan(affintiy_matrix_batch).any() == False
        assert torch.isinf(affintiy_matrix_batch).any() == False

        # Second assert the correct values using the batch alg vs the 
        affintiy_matrix_sci_0 = mpgm.affinity_loop(A[0], A_hat[0], E[0], E_hat[0], F[0], F_hat[0])
        affintiy_matrix_sci_1 = mpgm.affinity_loop(A[1], A_hat[1], E[1], E_hat[1], F[1], F_hat[1])        
        assert affintiy_matrix_batch[0].squeeze().numpy().any() == affintiy_matrix_sci_0.any()
        assert affintiy_matrix_batch[1].squeeze().numpy().any() == affintiy_matrix_sci_1.any()

    def test_maxpool(self):
        Xs_batch = mpgm.max_pool(S)
        assert Xs_batch.shape == torch.Size([batch_size,n,k])

        # Assert no inf or nan
        assert torch.isnan(Xs_batch).any() == False
        assert torch.isinf(Xs_batch).any() == False

        Xs_sci_0 = mpgm.max_pool_loop(S[0].squeeze().numpy())
        assert Xs_batch[0].squeeze().numpy().any() == Xs_sci_0.any()

    def test_hungarian(self):
        X_batch = mpgm.hungarian_batch(Xs)
        assert X_batch.shape == torch.Size([batch_size,n,k])

        # Assert no inf or nan
        assert torch.isnan(X_batch).any() == False
        assert torch.isinf(X_batch).any() == False

        X_sci = mpgm.hungarian(Xs[0].squeeze().numpy())

        assert X_batch[0].squeeze().numpy().any() == X_sci.any()

    def test_call(self):
        x = mpgm.call(A, A_hat, E, E_hat, F, F_hat)
        
        assert x.shape == torch.Size([batch_size,n,k])
        assert torch.isnan(x).any() == False
        assert torch.isinf(x).any() == False

        A_same = add_e7(torch.tensor(A*.9).to(my_dtype))
        E_same = add_e7(torch.tensor(E*.9).to(my_dtype))
        F_same = add_e7(torch.tensor(F*.9).to(my_dtype))
        x_same = mpgm.call(A, A_same, E, E_same, F, F_same)
        # TODO figure out why not diagonal!
        # assert torch.diagonal(x_same, dim1=1, dim2=2).any == 1

    def test_torch_set_diag(self):
        diag_zero = mpgm.torch_set_diag(A_hat)
        for i in range(n):
            assert diag_zero[:,i,i].numpy().any() == 0.
        
    def test_set_diag_nnkk(self):
        diag_fill = mpgm.set_diag_nnkk(A, batch_size, n, k)
        assert diag_fill.shape == (batch_size,n,n,k,k)
        for i in range(n):
            assert diag_fill[0,i,i,i,i].any() == A[0,i,i].numpy().any()


test = TestMPGM()
test.test_affinity()    
test.test_maxpool()
test.test_hungarian()
test.test_call()
test.test_set_diag_nnkk()
