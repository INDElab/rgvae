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


def original_gvae_loss(target, prediction, l_A=1., l_E=1., l_F=1., zero_diag: bool=True):
    """
    Loss function using max-pooling graph matching as describes in the GraphVAE paper.
    This loss expects A_hat to have sigmoid and E_hat and F_hat to have softmax probabilities.
    Meaning there can be several nodes and edges but only one edge and node attribute.
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
    log_p_F = (1/n) * torch.sum(torch.log(no_zero(torch.sum(F * F_hat_t, -1))), (-1)).unsqueeze(-1)

    # log_p_E
    log_p_E = ((1/(torch.norm(A, p=1, dim=[-2,-1])-n)) * torch.sum(torch.sum(torch.log(no_zero(E * E_hat_t)), -1) * mask, (-2,-1))).unsqueeze(-1)

    log_p = l_A * log_p_A + l_F * log_p_F + l_E * log_p_E
    return log_p

def mpgm_loss(target, prediction, l_A=1., l_E=1., l_F=1., zero_diag: bool=True):
    """
    Modification of the loss function described in the GraphVAE paper.
    The difference is, we treat A and E the same as both are sigmoided and F stays as it is softmaxed.
    This way we can have multiple edge attributes.
    The node attribute matrix is used to index the nodes, therefore the softmax.
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
    E_t = torch_batch_dot_v2(torch_batch_dot_v2(X, E, 1, 1, [bs,n,k,d_e]), X, -2, 1, [bs,k,k,d_e])    # target shape is (bs,k,k,d_e)
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
    log_p_F = (1/n) * torch.sum(torch.log(no_zero(torch.sum(F * F_hat_t, -1))), (-1)).unsqueeze(-1)

    # log_p_E
    e_norm = (1/(torch.norm(A, p=1, dim=[-2,-1])-n))
    log_p_E = (e_norm * torch.sum(torch.sum(E_t * torch.log(E_hat) + (1 - E_t) * torch.log(1 - E_hat), -1) * mask, (-2,-1))).unsqueeze(-1)

    log_p = l_A * log_p_A + l_F * log_p_F + l_E * log_p_E
    return log_p

def log_normal_pdf(sample, mean, logvar, raxis=1):
    """
    Function to numerically calculate the KL divergence. Not sure what they use the sample and log2pi for.
    Source: https://www.tensorflow.org/tutorials/generative/cvae
    """
    log2pi = torch.log(torch.ones_like(mean) * (2. * np.pi))
    return (torch.sum(-.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi), raxis)).unsqueeze(-1)

def kl_divergence(mean, logvar, raxis=1):
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
