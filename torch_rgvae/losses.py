"""
Collection of loss functions.
"""

from graph_matching.MPGM import MPGM
from utils import *


def graph_BCEloss(target, prediction, l_A=1., l_E=1., l_F=1.):
    """
    Binary cross entropy loss function for the predicted graph. Each matrix is taken into account seperatly.
    Args:
        target: list of the 3 target matrices A, E, F.
        prediction: list of the 3 predicted matrices A_hat, E_hat, F_hat.
        l_A: weight for BCE of A
        l_E: weight for BCE of E
        l_F: weight for BCE of F
    """

    # Cast target vectors to tensors.
    A, E, F = target
    A, E, F = torch.tensor(A * 1.), torch.tensor(E * 1.), torch.tensor(F * 1.)
    A_hat, E_hat, F_hat = prediction

    # Match number of nodes
    bce = torch.nn.BCELoss()
    loss = l_A*bce(A_hat, A) + l_E*bce(E_hat, E) + l_F*bce(F_hat, F)
    return loss


def mpgm_loss(target, prediction, l_A=1., l_E=1., l_F=1., zero_diag: bool=True):
    """
    Loss function using max-pooling graph matching as describes in the GraphVAE paper.
    Lets see if backprop works. Args obviously the same as above!
    Args:
        target: list of the 3 target matrices A, E, F.
        prediction: list of the 3 predicted matrices A_hat, E_hat, F_hat.
        l_A: weight for BCE of A
        l_E: weight for BCE of E
        l_F: weight for BCE of F
        zero_diag: if to zero out the diagonal in log_A term_3 and log_E.
    """
    A, E, F = target

    A_hat, E_hat, F_hat = prediction
    bs = A.shape[0]
    n = A.shape[1]
    k = A_hat.shape[1]
    d_e = E.shape[-1]

    # Cast target vectors to tensors.
    A = torch.tensor(A * 1.)
    E = torch.tensor(E * 1.)
    F = torch.tensor(F * 1.)

    mpgm = MPGM()
    X = mpgm.call(A, A_hat.detach(), E, E_hat.detach(), F, F_hat.detach())

    # This is the loss part from the paper:s
    A_t = torch.transpose(X, 2, 1) @ A @ X     # shape (bs,k,n)
    E_hat_t = torch_batch_dot_v2(torch_batch_dot_v2(X, E_hat, -1, 1, [bs,n,k,d_e]), X, -2, 1, [bs,n,n,d_e])    # target shape is (bs,n,n,d_e)
    F_hat_t = torch.matmul(X, F_hat)

    term_1 = (1/k) * torch.sum(torch.diagonal(A_t, dim1=-2, dim2=-1) * torch.log(torch.diagonal(A_hat, dim1=-2, dim2=-1)), -1, keepdim=True)
    A_t_diag = torch.diagonal(A_t, dim1=-2, dim2=-1)
    A_hat_diag = torch.diagonal(A_hat, dim1=-2, dim2=-1)
    term_2 = (1/k) * torch.sum((torch.ones_like(A_t_diag) - A_t_diag) * torch.log((torch.ones_like(A_hat_diag) - A_hat_diag)), -1, keepdim=True)

    """
    Thought: Lets compare w against w/o the zeroing out diagonal and see what happens.
    """
    # log_p_A part. Split in multiple terms for clarity.
    term_31 = A_t * torch.log(A_hat)
    term_32 = (1. - A_t) * torch.log(1. - A_hat)
    # Zero diagonal mask:
    mask = torch.ones_like(term_32)
    if zero_diag:
        ind = np.diag_indices(mask.shape[-1])
        mask[:,ind[0], ind[1]] = 0
    term_3 = (1/(k*(k-1))) * torch.sum((term_31 + term_32) * mask, [1,2]).unsqueeze(-1)
    log_p_A = term_1 + term_2 + term_3

    # log_p_F   
    F_nozero = torch.sum(F * F_hat_t, -1)
    F_nozero[F_nozero == 0] = 1.
    log_p_F = (1/n) * torch.sum(torch.log(F_nozero), -1).unsqueeze(-1)

    # log_p_E
    E_nozero = torch.sum(E * E_hat_t, -1)
    E_nozero[E_nozero==0] = 1.
    nor = (torch.norm(A, p='fro', dim=[-2,-1])-n)
    log_p_E = ((1/nor) * torch.sum(torch.log(E_nozero) * mask, (-2,-1))).unsqueeze(-1)

    log_p = l_A * log_p_A + l_F * log_p_F + l_E * log_p_E
    return log_p


def log_normal_pdf(sample, mean, logvar, raxis=1):
    """
    Function to numerically calculate the KL divergence. Not sure what they use the sample and log2pi for.
    Source: https://www.tensorflow.org/tutorials/generative/cvae
    """
    log2pi = torch.log(torch.ones_like(mean) * (2. * np.pi))
    return (torch.sum(-.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi), raxis)).unsqueeze(-1)

def KL_divergence(mean, logvar, raxis=1):
    """
    KL divergence between N(mean,std) and the standard normal N(0,1).
    Args:
        mean: mean of a normal dist.
        logvar: log variance (log(std**2)) of a normal dist.
    Returns Kl divergence in batch shape.
    """
    kl_term = 1/2 * torch.sum((logvar.exp() + mean.pow(2) - logvar - 1), dim=raxis)
    return kl_term.unsqueeze(-1)

def std_loss(prediction, l_A=1., l_E=1., l_F=1.):
    """
    This loss function pushes the model to generate more certain prediction by penalizing low variance.
    Args:
        predition: the torch_rgvae generated probabilistic output.
        l_*: weights for the matrices
    """
    A_hat, E_hat, F_hat = prediction
    std = l_A * torch.std(A_hat, dim=[-2,-1]) + l_E * torch.std(E_hat, dim=[-3,-2,-1]) + l_F * torch.std(F_hat, dim=[-2,-1])

    return  torch.mean(torch.log(std**2))**2

def sanity_check(sample, n: int, e: int):
    """
    Function to monitor the sanity logic of the prediction.
    Sanity 1: Model should predict graphs with the same amount of nodes as the target graph.
    Sanity 2: Model should predict graphs with the same amount of edges as the target graph.
    Sanity 3: Model should only predict edge attributes were it also predicts edges.
    Args:
        sample: A binarized prediction sample.
        n: number of nodes in target graph.
        e: number of edges in target graph.
    Returns:
        The 3 sanities in percentage.
    """
    A, E, F = sample
    
    # Sanity 1
    A_check = A.numpy()
    A_check = A_check[~np.all(A_check == 0, axis=1)]
    A_check = np.delete(A_check, np.where(~A_check.any(axis=1))[0], axis=0)
    k = A_check.shape[np.argmax(A_check.shape)] * 1.
    if k <= n:
        sanity_1 = k/n
    else:
        sanity_1 = 1 - (k-n)/n
    
    # Sanity 2
    e_check = np.sum(A_check)
    if e_check <= e:
        sanity_2 = e_check/e
    else:
        sanity_2 = 1 - (e_check-e)/e

    # Sanity 3
    E_check = torch.sum(E, -1)
    E_check[E_check > 0] = 1.
    zero_check = torch.zeros_like(A)
    zero_check[A == E_check] = 1
    sanity_3 = (torch.sum(zero_check)/(n**2)).item() 

    return sanity_1 * 100, sanity_2 * 100, sanity_3 * 100
