"""
Pytorch implementation of the max-pooling graph matching algorithm.
"""
import networkx as nx
import numpy as np
from numpy import array
from scipy.optimize import linear_sum_assignment
import torch
from munkres import Munkres, print_matrix, make_cost_matrix
from utils import *


class MPGM():
    def __init__(self):
        pass

    def call(self, A, A_hat, E, E_hat, F, F_hat):
        """
        Call the entire max_pooling algorithm.
        Input are the target and prediction matrices.
        Output is the discrete X matrix.
        """
        S = self.affinity(A, A_hat, E, E_hat, F, F_hat)
        X_star = self.max_pool(S)
        X = self.hungarian_batch(X_star)
        return X

    def call_test(self, A, A_hat, E, E_hat, F, F_hat):
        """
        A test run, does not work with batches. 1 to 1 implementation of the paper.
        Use this to verify your results if you decide to play around with the batch code.
        """
        S = self.affinity_loop(A, A_hat, E, E_hat, F, F_hat)
        X_star = self.max_pool_loop(S)
        X = self.hungarian(X_star)
        return X

    def ident_matching_nk(bs, n, k):
        # Returns ... not sure anymore

        X = torch.zeros([bs, n, k])
        for i in range(min(k,n)):
            X[:,i,i] = 1
        return X
    
    def torch_set_diag(self, t, filler=0.):
        """
        Pytorch fix of fill_diagonal for batches.
        The original function fill_diagonal_ only takes inputs with same dimensions. This makes it unsuited for batches.
        By reshaping it to two dimensions only we work around.
        """
        t_shape = t.shape
        t = torch.reshape(t, (-1, t_shape[-1])).fill_diagonal_(filler, wrap=True)
        return torch.reshape(t, t_shape)

    def set_diag_nnkk(self, S2, bs, n, k):
        """
        Returns zero matrix of shape (bs,n,n,k,k) with the (n.n) and (k,k) diagonal set as in S2.
        Input is the S2 (bs,n,k) diagonals.
        TODO this is not differentiable!!! - no need to!
        """
        X = np.zeros([bs,n,n,k,k])
        for i in range(n):
            the_diag = torch.diag_embed(S2[:,i,:])
            X[:,i,i,:,:] = the_diag.detach().numpy()
        return X
    

    def zero_diag_nnkk(self, bs, n, k, inverse=False):
        """
        Returns zero mask for (nn)  diagonal of a (bs,n,n,k,k) matrix.
        Input obvsl (bs,n,n,k,k)
        If inverse we inverse the mask.
        """
        X = np.ones([bs,n,n,k,k])
        if inverse:
            X = np.zeros([bs,n,n,k,k])
        for i in range(n):
            X[:,i,i,:,:] = 0
            if inverse:
                X[:,i,i,:,:] = 1
        return torch.tensor(X)

    def affinity(self, A, A_hat, E, E_hat, F, F_hat):
        """
        Let's make some dimensionalities clear first:
            A: nxn
            E: nxnxd_e
            F: nxd_n
            A_hat: kxk
            E_hat: kxkxd_e
            F_hat: kxd_n
        In an ideal world the target dimension n and the predictions dim k are the same.
        The other two dimensions are node and edge attributes. All matrixes come in batches, which is the first dimension.
        Now we are going to try to solve this with matrix multiplication, for-loops are a no-go.
        My first shot would be to implement this formula without respecting the constrains:
        S((i, j),(a, b)) = (E'(i,j,:)E_hat(a,b,:))A(i,j)A_hat(a,b)A_hat(a,a)A_hat(b,b) [i != j ∧ a != b] + (F'(i,:)F_hat(a,:))A_hat(a,a) [i == j ∧ a == b]
        And later mask the constrained entries with zeros.
        TODO To test it we could run a single sample and compare the loop and the matmul output.
        """
        n = A.shape[1]
        self.n = n
        k = A_hat.shape[1]
        self.k = k
        bs = A.shape[0]     # bs stands for batch size, just to clarify.
        self.bs = bs

        F = F.to(float)
        A = A.to(float)
        E = E.to(float)

        F_hat_t = torch.transpose(F_hat, 1, 2)
        A_hat_diag = (torch.diagonal(A_hat,dim1=-2,dim2=-1)).unsqueeze(-1)
        A_hat_diag_t = torch.transpose(A_hat_diag, 2, 1)

        S11 = torch_batch_dot(E, E_hat, 3, 3)   # We aim for shape (batch_s,n,n,k,k).

        # Now we need to get the second part into shape (batch_s,n,n,k,k).
        S121 = A_hat_diag @ A_hat_diag_t
        # This step masks out the (a,b) diagonal. TODO: Make it optional.
        S12 = self.torch_set_diag(S121).unsqueeze(-1)

        A = A.unsqueeze(-1)
        S13 = torch_batch_dot(A, S12, -1, -1)

        # Pointwise multiplication of E and A matrices
        S1 = S11 * S13

        S21 = A_hat_diag.expand(bs,k,n)  # This repeats the input vector to match the F shape.
        S2 = torch.matmul(F, F_hat_t) * torch.transpose(S21, 2, 1)      # I know this looks weird but trust me, I thought this through!
        S2 = self.set_diag_nnkk(S2, bs, n, k)    # This puts the values on the intended diagonal to match the shape of S

        # This zero masks the (n,n) diagonal
        S1 = S1 * self.zero_diag_nnkk(bs, n, k)

        return S1 + S2

    def affinity_loop(self, A, A_hat, E, E_hat, F, F_hat):
        # We are going to iterate over pairs of (a,b) and (i,j)
        # np.nindex is going to make tuples to avoid two extra loops.
        ij_pairs = list(np.ndindex(A.shape))
        ab_pairs = list(np.ndindex(A_hat.shape))
        n = A.shape[0]
        self.n = n
        k = A_hat.shape[0]
        self.k = k
        # create en empty affinity matrix.
        S = np.empty((n,n,k,k))

        # Now we start filling in the S matrix.
        for (i, j) in ij_pairs:
            for (a, b) in ab_pairs:
                # OMG this loop feels sooo wrong!
                if a != b and i != j:
                    A_scalar = A[i,j] * A_hat[a,b] * A_hat[a,a] * A_hat[b,b]
                    S[i,j,a,b] = np.matmul(np.transpose(E[i,j,:]), E_hat[a,b,:]) * A_scalar
                    del A_scalar
                elif a == b and i == j:
                    S[i,j,a,b] = np.matmul(np.transpose(F[i,:]), F_hat[a,:]) * A_hat[a,a]
                else:
                    # For some reason the similarity between two nodes for the case when one node is on the diagonal is not defined.
                    # We will set these points to zero until we find an better solution. 
                    S[i,j,a,b] = 0.
        return S
    
    def max_pool(self, S, n_iterations: int=300):
        """
        The famous Cho max-pooling in matrix multiplication style.
        Xs: X_star meaning X in continuos space.
        """
        # Just a crazy idea, but what if we flatten the X (n,k) matrix so that we can take the dot product with S (n,flat,K).
        Xs = torch.rand([self.bs, self.n, self.k])
        self.Xs = Xs
        S = torch.reshape(S, [S.shape[0],S.shape[1],S.shape[-2],-1])
        for n in range(n_iterations):
            Xs = torch.reshape(Xs, [self.bs,-1]).unsqueeze(1).unsqueeze(-1)
            SXs = torch.matmul(S,Xs).squeeze()
            xnorm = torch.norm(SXs, p='fro', dim=[-2,-1])
            Xs = (SXs / xnorm.unsqueeze(-1).unsqueeze(-1))
        return Xs

    def max_pool_loop(self, S, n_iterations: int=300):
        """
        Input: Affinity matrix
        Output: Soft assignment matrix
        Args:
            S (np.array): Float affinity matrix of size (k,k,n,n)
            n_iterations (int): Number of iterations for calculating X
        """
        # The magic happens here, we are going to iteratively max pool the S matrix to get the X matrix.
        # We initiate the X matrix random uniform.
        k = self.k
        n = self.n
        if self.Xs is:
            X = self.Xs.squeeze().numpy()
        else:
            X = np.random.uniform(size=(n,k))
        # make pairs
        ia_pairs = list(np.ndindex(X.shape))

        #Just to make sure we are not twisting things. note: shape = dim+1
        assert ia_pairs[-1] == (n-1,k-1), 'Dimensions should be ({},{}) but are {}'.format(n-1,k-1,ia_pairs[-1])

        #loop over iterations and paris
        for it in range(n_iterations):
            for (i, a) in ia_pairs:
                # TODO the paper says argmax and sum over the 'neighbors' of node pair (i,a).
                # My interpretation is that when there is no neighbor the S matrix will be zero, there fore we still use j anb b in full rage.
                # Second option would be to use a range of [i-1,i+2].
                # The first term max pools over the pairs of edge matches (ia;jb).
                de_sum = np.sum([np.argmax(X[j,:] @ S[i,j,a,:]) for j in range(k)])
                # In the next term we only consider the node matches (ia;ia).
                X[i,a] = X[i,a] * S[i,i,a,a] + de_sum
            # Normalize X to range [0,1].
            X = X * 1./np.linalg.norm(X)
        return X

    def hungarian(self, X_star, cost: bool=False):
        """ 
        Apply the hungarian or munkres algorithm to the continuous assignment matrix.
        The output is a discrete similarity matrix.
        Are we working with a cost or a profit matrix???
        Args:
            X_star: numpy array matrix of size n x k with elements in range [0,1]
            cost: Boolean argument if to assume the input is a profit matrix and to convert is to a cost matrix or not.
        """
        m = Munkres()
        if cost:
            X_star = make_cost_matrix(X_star)
        # Compute the indexes for the matrix for the lowest cost path.        
        indexes = m.compute(X_star.copy())

        # Now mast these indexes with 1 and the rest with 0.
        X = np.zeros_like(X_star, dtype=int)
        for idx in indexes:
            X[idx] = 1
        return X

    def hungarian_batch(self, Xs):
        X = Xs.numpy().copy()
        for i in range(X.shape[0]):
            # We are always given square Xs, but some may have unused columns (ground truth nodes are not there), so we can crop them for speedup. It's also then equivalent to the original non-batched version.
            row_ind, col_ind = linear_sum_assignment(X[i])
            M = np.zeros(X[i].shape, dtype=float)
            M[row_ind, col_ind] = 1.
            X[i] = M
        return torch.tensor(X)



if __name__ == "__main__":

    my_dtype = torch.float64
    torch.set_default_dtype(my_dtype)

    # Let's define some dimensions :)
    n = 2
    k = 2
    d_e = 2
    d_n = 4

    batch_size = 2
    seed = 11
    # Generation of random test graphs. The target graph is discrete and the reproduced graph probabilistic.
    torch.manual_seed(seed)
    np.random.seed(seed=seed)
    A = np.random.randint(2, size=(batch_size,n,n))
    E = np.random.randint(2, size=(batch_size,n,n,d_e))
    F = np.random.randint(2, size=(batch_size,n,d_n))
    A_hat = torch.randn((batch_size,k,k))
    E_hat = torch.randn((batch_size,k,k,d_e))
    F_hat = torch.randn((batch_size,k,d_n))


    # Test the class, actually this should go in a test function and folder. Later...
    mpgm = MPGM()

    S = mpgm.affinity(A, A_hat, E, E_hat, F, F_hat)
    Xs = mpgm.max_pool(S)
    X = mpgm.hungarian_batch(Xs)
    print(Xs[0])
    S2 = mpgm.affinity_loop(np.squeeze(A[0]), np.squeeze(A_hat[0]), np.squeeze(E[0]), np.squeeze(E_hat[0]), np.squeeze(F[0]), np.squeeze(F_hat[0]))
    Xs2 = mpgm.max_pool_loop(S2)
    X2 = mpgm.hungarian(Xs2)
    print(Xs2)
