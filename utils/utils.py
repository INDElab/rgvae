"""
Utility functions.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from tqdm import tqdm


def linear_sum_assignment_with_inf(cost_matrix):
    """
    scipy linear sum assignment for cost matrices with inf or -inf.
    Source: https://github.com/scipy/scipy/issues/6900
    """

    cost_matrix = np.asarray(cost_matrix)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        values = cost_matrix[~np.isinf(cost_matrix)]
        m = values.min()
        M = values.max()
        n = min(cost_matrix.shape)
        # strictly positive constant even when added
        # to elements of the cost matrix
        positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
        if max_inf:
            place_holder = (M + (n - 1) * (M - m)) + positive
            cost_matrix[np.isinf(cost_matrix)] = place_holder
        if min_inf:
            place_holder = (m + (n - 1) * (m - M)) - positive
            cost_matrix[np.isinf(cost_matrix)] = place_holder

    return linear_sum_assignment(cost_matrix)

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


def sum_sparse(indices, values, size, row_normalisation=True, device='cpu'):
    """
    Sum the rows or columns of a sparse matrix, and redistribute the
    results back to the non-sparse row/column entries
    Arguments are interpreted as defining sparse matrix.

    Source: https://github.com/pbloem/gated-rgcn/blob/1bde7f28af8028f468349b2d760c17d5c908b58b/kgmodels/util/util.py#L304
    """

    assert len(indices.size()) == len(values.size()) + 1

    k, r = indices.size()

    if not row_normalisation:
        # Transpose the matrix for column-wise normalisation
        indices = torch.cat([indices[:, 1:2], indices[:, 0:1]], dim=1)
        size = size[1], size[0]

    ones = torch.ones((size[1], 1), device=device)
    if device == 'cuda':
        values = torch.cuda.sparse.FloatTensor(indices.t(), values, torch.Size(size))
    else:
        values = torch.sparse.FloatTensor(indices.t(), values, torch.Size(size))
    sums = torch.spmm(values, ones)
    sums = sums[indices[:, 0], 0]

    return sums.view(k)


def generate_inverses(triples, num_rels):
    """ Generates nverse relations """

    # Swap around head and tail. Create new relation ids for inverse relations.
    inverse_relations = torch.cat([triples[:, 2, None], triples[:, 1, None] + num_rels, triples[:, 0, None]], dim=1)
    assert inverse_relations.size() == triples.size()

    return inverse_relations


def generate_self_loops(triples, num_nodes, num_rels, self_loop_keep_prob, device='cpu'):
    """ Generates self-loop triples and then applies edge dropout """

    # Create a new relation id for self loop relation.
    all = torch.arange(num_nodes, device=device)[:, None]
    id  = torch.empty(size=(num_nodes, 1), device=device, dtype=torch.long).fill_(2*num_rels)
    self_loops = torch.cat([all, id, all], dim=1)
    assert self_loops.size() == (num_nodes, 3)

    # Apply edge dropout
    mask = torch.bernoulli(torch.empty(size=(num_nodes,), dtype=torch.float, device=device).fill_(
        self_loop_keep_prob)).to(torch.bool)
    self_loops = self_loops[mask, :]

    return torch.cat([triples, self_loops], dim=0)


def stack_matrices(triples, num_nodes, num_rels, vertical_stacking=True, device='cpu'):
    """
    Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all
    relations are stacked vertically).
    """
    assert triples.dtype == torch.long

    r, n = num_rels, num_nodes
    size = (r * n, n) if vertical_stacking else (n, r * n)

    fr, to = triples[:, 0], triples[:, 2]
    offset = triples[:, 1] * n
    if vertical_stacking:
        fr = offset + fr
    else:
        to = offset + to

    indices = torch.cat([fr[:, None], to[:, None]], dim=1).to(device)

    assert indices.size(0) == triples.size(0)
    assert indices[:, 0].max() < size[0], f'{indices[0, :].max()}, {size}, {r}'
    assert indices[:, 1].max() < size[1], f'{indices[1, :].max()}, {size}, {r}'

    return indices, size


def block_diag(m):
    """
    Source: https://gist.github.com/yulkang/2e4fc3061b45403f455d7f4c316ab168
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    """

    device = 'cuda' if m.is_cuda else 'cpu'  # Note: Using cuda status of m as proxy to decide device

    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    dim = m.dim()
    n = m.shape[-3]

    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]

    m2 = m.unsqueeze(-2)

    eye = attach_dim(torch.eye(n, device=device).unsqueeze(-2), dim - 3, 1)

    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))

