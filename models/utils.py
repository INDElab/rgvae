"""
Utility functions.
"""

import numpy as np
import tensorflow as tf

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



def replace_nan(t):
    """
    Function to replace NaNs.
    """
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)

def add_e7(t):
    """
    Function to add a very small value to each element, to avoid inf errors when taking the logarithm.
    """
    return t + tf.ones_like(t) * 1e-7

    



if __name__ == "__main__":
    print(mk_random_graph(3,2,2,2))
    print(mk_random_graph_ds(3,2,2,2,2)[0])
