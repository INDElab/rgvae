"""
Utility functions.
"""
import numpy as np
import torch



def mk_random_graph(n: int, d_e: int, d_n: int, batch_size: int=1, target: bool=True):
    """
    This function creates a batch of random graphs.
    Each graph is made of an adjacency, an edge-attribute and a node-attribute matrix.
    If we choose to generate a target graph, the graph values are deterministic.
    Otherwise we are generating a prediction graph with continuous values.
    returns a list of 3 numpy matrices. TODO: make tf an option?
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

def mk_random_graph_ds(n: int, d_e: int, d_n: int, batches: int=1, batch_size: int=1,target: bool=True):
    """
    Forbatches.
    """
    ds = list()
    for i in range(batches):
        ds.append(mk_random_graph(n,d_e,d_n,batch_size,target))
    return ds

def torch_batch_dot(M1, M2, dim1, dim2):
    """
    Torch implementation of the batch dot matrix multiplication.
    """
    M1_shape = M1.shape
    M2_shape = M2.shape
    bs = M1_shape[0]
    M3 = torch.matmul(M1.view(bs,-1,M1_shape[dim1]),M2.view(bs,M2_shape[dim2],-1)).view(bs,M1_shape[1],M1_shape[2],M2_shape[1],M2_shape[2])
    return M3

def replace_nan(t):
    """
    Function to replace NaNs.
    """
    return torch.where(torch.is_nan(t), torch.zeros_like(t), t)

def add_e7(t):
    """
    Function to add a very small value to each element, to avoid inf errors when taking the logarithm.
    """
    return t + torch.ones_like(t) * 1e-7

    



if __name__ == "__main__":
    print(mk_random_graph(3,2,2,2))
    print(mk_random_graph_ds(3,2,2,2,2)[0])
