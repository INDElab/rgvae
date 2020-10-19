"""
Pytorch implementation of the max-pooling graph matching algorithm.
"""
import time
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
    
    def torch_set_diag(self, t, filler=0.):
        """
        Pytorch fix of fill_diagonal for batches.
        We assume the input tensor has shape (bs,n,n).
        """
        t_return = torch.tensor(t)
        ind = np.diag_indices(t.shape[-1])
        t_return[:,ind[0], ind[1]] = torch.ones(t.shape[-1]) * filler
        return t_return

    def set_diag_nnkk(self, S2, bs, n, k):
        """
        Returns zero matrix of shape (bs,n,n,k,k) with the (n.n) and (k,k) diagonal set as in S2.
        Input is the S2 (bs,n,k) diagonals.
        """
        X = np.zeros([bs,n,n,k,k])
        for i in range(n):
            the_diag = torch.diag_embed(S2[:,i,:])
            X[:,i,i,:,:] = the_diag.detach().numpy()
        return X

    def affinity(self, A, A_hat, E, E_hat, F, F_hat):
        """
        Let's make some dimensionalities clear first (w/o batch dim):
            A: n,n
            E: n,n,d_e
            F: n,d_n
            A_hat: k,k
            E_hat: k,k,d_e
            F_hat: k,d_n
        In an ideal world the target dimension n and the predictions dim k are the same.
        The other two dimensions are node and edge attributes. All matrixes come in batches, which is the first dimension.
        Now we are going to try to solve this with matrix multiplication, for-loops are a no-go.
        My first shot would be to implement this formula without respecting the constrains:
        S((i, j),(a, b)) = (E'(i,j,:)E_hat(a,b,:))A(i,j)A_hat(a,b)A_hat(a,a)A_hat(b,b) [i != j ∧ a != b] + (F'(i,:)F_hat(a,:))A_hat(a,a) [i == j ∧ a == b]
        And later mask the constrained entries with zeros.
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

        A_hat_diag = (torch.diagonal(A_hat,dim1=-2,dim2=-1)).unsqueeze(-1)
        E_norm = torch.torch.norm(E,p=1,dim=-1,keepdim=True)  # Division by the norm since our model can have multiple edge attributes vs. one-hot
        E_norm[E_norm == 0.] = 1.       # Otherwise we get nans
        E_ijab = torch_batch_dot(E/E_norm, E_hat, 3, 3)   # We aim for shape (batch_s,n,n,k,k).

        A_ab = A_hat * self.torch_set_diag(torch_batch_dot_v2(A_hat_diag,A_hat_diag, -1, -1, (bs,k,k)))
        A_ijab = torch_batch_dot_v2((self.torch_set_diag(A)).unsqueeze(-1),A_ab.unsqueeze(-1), -1, -1, (bs,n,n,k,k))

        A_aa = torch.bmm(torch.ones((bs,n,1)), torch.transpose(A_hat_diag,1,2))
        F_ia = torch.matmul(F, torch.transpose(F_hat, 1, 2))

        # S = E_ijab * A_ijab + self.set_diag_nnkk(F_ia * A_aa, bs, n, k)
        # assert torch.isnan(S).any() == False

        S_iaia = F_ia * A_aa
        S_iajb = E_ijab * A_ijab #+ self.set_diag_nnkk(S_iaia, bs, n, k)
        return (S_iajb, S_iaia)

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
        S_iajb, S_iaia = S
        Xs = torch.ones_like(S_iaia)
        self.Xs = Xs
        for n in range(n_iterations):
            Xs = Xs * S_iaia + torch.sum(torch.max(S_iajb * Xs.unsqueeze(1).unsqueeze(1),-1, out=None)[0],-1)
            Xs_norm = torch.norm(Xs, p='fro', dim=[-2,-1])
            Xs = (Xs / Xs_norm.unsqueeze(-1).unsqueeze(-1))
        return Xs
        

        # # Just a crazy idea, but what if we flatten the X (n,k) matrix so that we can take the dot product with S (n,flat,K).
        # Xs = torch.rand([self.bs, self.n, self.k])
        # self.Xs = Xs
        # S = torch.reshape(S, [S.shape[0],S.shape[1],S.shape[-2],-1])
        # for n in range(n_iterations):
        #     Xs = torch.reshape(Xs, [self.bs,-1]).unsqueeze(1).unsqueeze(-1)
        #     SXs = torch.matmul(S,Xs).squeeze()
        #     xnorm = torch.norm(SXs, p='fro', dim=[-2,-1])
        #     Xs = (SXs / xnorm.unsqueeze(-1).unsqueeze(-1))
        #     assert torch.isnan(Xs).any() == False
        # return Xs

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
        S_iajb, S_iaia = S
        k = self.k
        n = self.n
        if self.Xs is None:
            X = np.random.uniform(size=(n,k))
        else:
            # Using the first Xs from the batch, thus we can only compare the first matrix in the batch
            X = self.Xs[0].squeeze().numpy()
            
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
                de_sum = np.sum([np.argmax(X[j,:] @ S_iajb[i,j,a,:]) for j in range(n)])
                # In the next term we only consider the node matches (ia;ia).
                X[i,a] = X[i,a] * S_iaia[i,a] + de_sum
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
        X = Xs.clone().numpy()
        # Make it a cost matrix
        X = np.ones_like(X) - X
        for i in range(X.shape[0]):
            row_ind, col_ind = linear_sum_assignment(X[i])
            M = np.zeros(X[i].shape, dtype=float)
            M[row_ind, col_ind] = 1.
            X[i] = M
        X = torch.tensor(X)
        return X

