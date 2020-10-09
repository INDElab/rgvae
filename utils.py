"""
Utility functions.
"""
import numpy as np
import torch
from tqdm import tqdm


def no_zero(t):
    """
    This function replaces all zeros in a tensor with ones.
    This allows us to take the logarithm and then sum over all values in the matrix.
    Args:
        t: tensor to be replaced
    returns:
        t: tensor with ones instead of zeros.
    """
    t[t==0] = 1.
    return t

def check_adj_logic(sample):
    """
    Checks if the generated sample adheres to the logic, that edge attributes can only exist where the adjacency matrix indicates an edge.
    Args:
        sample: A binomial sample of a predicted graph.
    Output:
        Not sure yet.
    """
    A, E, F = sample
    E_check = torch.sum(E, -1)
    E_check[E_check > 0] = 1.
    bool_check = A[A == E_check]
    print(A == E_check)

def mk_sparse_graph_ds(n: int, e: int, d_e: int, batch_size: int=1, batches: int=1):
    """
    Function to create random graph dataet in sparse matrix form.
    We generate the each subject (s), relation (r), object (o) vector seperate and then stack and permute them.
    Output shape is [batches*(bs,[s,r,o])].
    Args:
        n: number of nodes.
        e: number of edges between nodes.
        d_e: number of edge attributes.
        batch_size: well, the batch size.
        batches: optional for unicorn dust.
    """
    ds = list()
    for _ in range(batches):
        s = np.random.choice(n, (batch_size, e))
        r = np.random.choice(d_e, (batch_size, e))
        o = np.random.choice(n, (batch_size, e))
        ds.append(np.stack([s,r,o], axis=-1))
    return ds

def mk_cnstrnd_graph(n: int, e: int, d_e: int, d_n: int, batch_size: int=1, self_loop: bool=False):
    """
    Returns a random Graph constrained on the number of nodes and edges.
    Args:
        n: number of nodes. defines the shape of the adjacency matrix.
        e: number of edges, this is the constrain
        d_e: number of edge-attributes.
        d_n: number of node-attributes.
        batch_size: well.. the batch size.
        self_loop: Set the diagonal of the adj matrix to one.
    """
    lambda_choice = lambda x,y: np.random.choice(x, y, replace=False)
    a_choice = np.append(np.ones(e, dtype=int), np.zeros(n*n - e, dtype=int))
    A = np.vstack([lambda_choice(a_choice,n*n) for _ in range(batch_size)])
    A = A.reshape((batch_size,n,n))

    if self_loop:
        one_diag = np.eye(n, dtype=int)
        one_diag = np.tile(np.expand_dims(one_diag, axis=0), (batch_size, 1, 1))
        A = A + one_diag

    # The idea here is that an edge attribute can only exist where an edge is. Further if there is an edge we want at leat one attribute to be 1.
    E = np.zeros((batch_size,n,n,d_e), dtype=int)
    E[:,:,:,0] = A.copy()
    e_choice = np.append(np.ones(d_e, dtype=int), np.zeros(d_e-1, dtype=int))
    
    E[A==1,:] = np.vstack([lambda_choice(e_choice, d_e) for _ in range(batch_size*e)])

    f_choice = np.append(np.ones(1, dtype=int), np.zeros(d_n-1, dtype=int))

    F = np.eye(d_n)[np.random.choice(d_n,batch_size*n)].reshape((batch_size,n,d_n))
    return A, E, F

def mk_random_graph(n: int, d_e: int, d_n: int, batch_size: int=1, target: bool=True):
    """
    This function creates a random relation graph.
    Consisting of an adjacency, an edge-attribute and a node-attribute matrix.
    If we choose to generate a target graph, the graph values are deterministic.
    Otherwise we are generating a prediction graph with continuous values.
    returns a list of 3 numpy matrices. TODO: F = A + 3rd dim
    Args:
        n: number of nodes. defines the shape of the adjacency matrix.
        d_e: number of edge-attributes.
        d_n: number of node-attributes.
        batch_size: well.. the batch size.
        target: generates a target graph when True, a prediction graph otherwise.
    """
    if target:
        A = np.random.randint(2, size=(batch_size,n,n))
        E = np.random.randint(2, size=(batch_size,n,n,d_e))
        F = np.random.randint(2, size=(batch_size,n,d_n))
    else:
        A = np.random.normal(size=(batch_size,k,k))
        E = np.random.normal(size=(batch_size,k,k,d_e))
        F = np.random.normal(size=(batch_size,k,d_n))
    return (A, E, F)

def mk_graph_ds(n: int, d_e: int, d_n: int, e: int, constrained: bool=True, batches: int=1, batch_size: int=1,target: bool=True):
    """
    Forbatches.
    Args:
        n: number of nodes. defines the shape of the adjacency matrix.
        e: number of edges, if constrained.
        d_e: number of edge-attributes.
        d_n: number of node-attributes.
        batch_size: well.. the batch size.
        target: generates a target graph when True, a prediction graph otherwise.
    """
    ds = list()
    if constrained:
        for i in tqdm(range(batches), desc='Creating Dataset', total=batches):
            ds.append(mk_cnstrnd_graph(n,e,d_e,d_n,batch_size))
    else:
        for i in tqdm(range(batches), desc='Creating Dataset', total=batches):
            ds.append(mk_random_graph(n,d_e,d_n,batch_size,target))
    return ds

def torch_batch_dot(M1, M2, dim1, dim2):
    """
    Torch implementation of the batch dot matrix multiplication.
    Only for matrices of shape (bs,n,n,1) and (bs,k,k,1).
    Returns matrix of shape (bs,n,n,k,k).
    """
    M1_shape = M1.shape
    M2_shape = M2.shape
    bs = M1_shape[0]
    M3 = torch.matmul(M1.view(bs,-1,M1_shape[dim1]), M2.view(bs,M2_shape[dim2],-1)).view(bs,M1_shape[1],M1_shape[2],M2_shape[1],M2_shape[2])
    return M3

def torch_batch_dot_v2(M1, M2, dim1, dim2, return_shape):
    """
    Torch implementation of the batch dot matrix multiplication.
    Args:
        return_shape: The shape of the returned matrix, including batch size.
    """
    M1_shape = M1.shape
    M2_shape = M2.shape
    bs = M1_shape[0]
    M3 = torch.matmul(M1.view(bs,-1,M1_shape[dim1]), M2.view(bs,M2_shape[dim2],-1)).view(return_shape)
    return M3

def replace_nan(t):
    """
    Function to replace NaNs.
    """
    return torch.where(torch.isnan(t), torch.zeros_like(t), t)

def replace_inf(t):
    """
    Function to replace NaNs.
    """
    return torch.where(torch.isinf(t), torch.zeros_like(t), t)

def add_e7(t):
    """
    Function to add a very small value to each element, to avoid inf errors when taking the logarithm.
    """
    return t + torch.ones_like(t) * 1e-7

    



if __name__ == "__main__":
    print(mk_cnstrnd_graph(5,10,3,3,2))
    # print(mk_graph_ds(5,3,3,11,constrained=True,batches=400,batch_size=64)[0])

    # print(mk_sparse_graph_ds(5,10,3,2,2))
