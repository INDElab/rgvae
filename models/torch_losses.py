"""
Collection of loss functions.
"""

import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
from models.torch_max_pooling import MPGM
from models.utils import *


def graph_loss(target, prediction, l_A=1., l_E=1., l_F=1.):
    """
    Loss function for the predicted graph. It takes each matrix separately into account.
    Goal is to solve the permutation invariance.
    Args:
        A_hat: Predicted adjencency matrix.
        E_hat: Predicted edge-attribute matrix.
        F_hat: Predicted node-attribute matrix.
        A: Ground truth adjencency matrix.
        E: Ground truth edge-attribute matrix.
        F: Ground truth node-attribute matrix.
    """
    # Set weights for different parts of the loss function
    w1 = 0
    w2 = 2
    w3 = 2
    w4 = 1

    # Cast target vectors to tensors.
    A, E, F = target
    A_hat, E_hat, F_hat = prediction

    # Match number of nodes
    loss_n_nodes = torch.sqrt((torch.count_nonzero(A) - torch.count_nonzero(A_hat))**2)
    bce = torch.nn.BCELoss(from_logits=False)
    loss = w1*loss_n_nodes + w2*bce(A, A_hat) + w3*bce(E, E_hat) + w4*bce(F, F_hat)
    return loss


def mpgm_loss(target, prediction, l_A=1., l_E=1., l_F=1.):
    """
    Loss function using max-pooling graph matching as describes in the GraphVAE paper.
    Lets see if backprop works. Args obviously the same as above!
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

    # now comes the loss part from the paper:s
    A_t = torch.transpose(X, 2, 1) @ A @ X     # shape (bs,k,n)
    E_hat_t = torch_batch_dot_v2(torch_batch_dot_v2(X, E_hat, -1, 1, [bs,n,k,d_e]), X, -2, 1, [bs,n,n,d_e])    # target shape is (bs,n,n)
    F_hat_t = torch.matmul(X, F_hat)

    # To avoid inf or nan errors we add the smallest possible value to all elements.
    A_hat_4log = add_e7(A_hat)

    term_1 = (1/k) * torch.sum(torch.diagonal(A_t, dim1=-2, dim2=-1) * torch.log(torch.diagonal(A_hat, dim1=-2, dim2=-1)), -1, keepdim=True)
    A_t_diag = torch.diagonal(A_t, dim1=-2, dim2=-1)
    A_hat_diag = torch.diagonal(A_hat, dim1=-2, dim2=-1)
    term_2 = torch.sum((torch.ones_like(A_t_diag) - A_t_diag) * (torch.ones_like((A_hat_diag)) - torch.log(A_hat_diag)), -1, keepdim=True)

    """
    Thought: Lets skip the zeroing out diagonal and see what happens. This also blocks the backprop, afaik.
    """
    # term_31 = set_diag(A_t, tf.zeros_like(diag_part(A_t))) * set_diag(tf.math.log(A_hat_4log), tf.zeros_like(diag_part(A_hat)))
    term_31 = A_t * torch.log(A_hat)
    # term_31 = replace_nan(term_31)        # I LIKE NANs - said no one ever.

    # term_32 = tf.ones_like(A_t) - set_diag(A_t, tf.zeros_like(diag_part(A_t))) * tf.math.log(tf.ones_like(A_t) - set_diag(A_hat_4log, tf.zeros_like(diag_part(A_hat))))
    term_32 = torch.ones_like(A_t) - A_t * torch.log(torch.ones_like(A_t) - A_hat)
    term_32 = replace_nan(term_32)
    term_3 = (1/k*(1-k)) * torch.sum(term_31 + term_32, [1,2]).unsqueeze(-1)
    log_p_A = term_1 + term_2 + term_3

    # Man so many confusions: is the log over one or both Fs???
    log_p_F = (1/n) * torch.sum(torch.log(add_e7(torch.sum((F * F_hat_t), -1))), -1).unsqueeze(-1)

    ### TODO THIS IS WHERE THE BACKPROP CRASHES ###
    log_p_E = ((1/(torch.norm(A, p='fro', dim=[-2,-1])-n)) * torch.sum(torch.log(add_e7(torch.sum(E * E_hat_t, -1))), (-2,-1))).unsqueeze(-1)

    log_p = - l_A * log_p_A - l_F * log_p_F - l_E * log_p_E
    return log_p


def log_normal_pdf(sample, mean, logvar, raxis=1):
    # mean = torch.tensor(mean)
    # logvar = torch.tensor(logvar)
    log2pi = torch.log(torch.ones_like(mean) * (2. * np.pi))
    return (torch.sum(-.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi), raxis)).unsqueeze(-1)
