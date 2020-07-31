"""
Collection of loss functions.
"""

import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch_max_pooling import MPGM
from utils import *


def graph_loss(A, E, F, A_hat, E_hat, F_hat):
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
    A = torch.tensor(A)
    E = torch.tensor(E)
    F = torch.tensor(F)

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
    n = A.shape[1]
    k = A_hat.shape[1]

    # Cast target vectors to tensors.
    A = torch.tensor(A, dtype=torch.float)
    E = torch.tensor(E, dtype=torch.float)
    F = torch.tensor(F, dtype=torch.float)

    mpgm = MPGM()
    X = mpgm.call(A, A_hat, E, E_hat, F, F_hat)

    # now comes the loss part from the paper:
    A_t = torch.transpose(X, 2, 1) @ A @ X     # shape (bs,k,n)
    E_hat_t = torch.transpose(batch_dot(batch_dot(X, E_hat, -1, 1), X, (-2,1)), 3, 2)
    F_hat_t = torch.matmul(X, F_hat)

    # To avoid inf or nan errors we add the smallest possible value to all elements.
    A_hat_4log = add_e7(A_hat)

    term_1 = (1/k) * torch.sum(torch.diagonal(A_t) * torch.log(torch.diagonal(A_hat)), -1, keepdim=True)


    term_2 = torch.sum((torch.ones_like(torch.diagonal(A_t)) - torch.daigonal(A_t)) * (torch.ones_like(torch.diagonal(A_hat)) - torch.log(diag_part(A_hat)))), -1, keepdim=True) 
    
    # TODO unsure if (1/(k*(1-k))) or ((1-k)/k) ??? Also the second sum in the paper is confusing. I am going to interpret it as matrix multiplication and sum over all elements.
    b = torch.diagonal(A_t)
    """
    Thought: Lets skip the zeroing out diagonal and see what happens. This also blocks the backprop, afaik.
    """
    # term_31 = set_diag(A_t, tf.zeros_like(diag_part(A_t))) * set_diag(tf.math.log(A_hat_4log), tf.zeros_like(diag_part(A_hat)))
    term_31 = A_t * torch.log(A_hat)
    term_31 = replace_nan(term_31)        # I LIKE NANs - said no one ever.

    # term_32 = tf.ones_like(A_t) - set_diag(A_t, tf.zeros_like(diag_part(A_t))) * tf.math.log(tf.ones_like(A_t) - set_diag(A_hat_4log, tf.zeros_like(diag_part(A_hat))))
    term_32 = tf.ones_like(A_t) - A_t * torch.log(tf.ones_like(A_t) - A_hat)
    term_32 = replace_nan(term_32)
    term_3 = (1/k*(1-k)) * tf.expand_dims(tf.math.reduce_sum(term_31 + term_32, [1,2]), -1)
    log_p_A = term_1 + term_2 + term_3

    # Man so many confusions: is the log over one or both Fs???
    log_p_F = (1/n) * torch.sum(torch.log((torch.sum((F * F_hat_t), -1), -1).unsqueeze(-1))

    log_p_E = ((1/(torch.norm(A, p='fro', dim=[-2,-1])-n)) * torch.sum(torch.log(torch.sum(E * E_hat_t, -1)), (-2,-1))).unsqueeze(-1)

    log_p = - l_A * log_p_A - l_F * log_p_F - l_E * log_p_E
    return log_p


def log_normal_pdf(sample, mean, logvar, raxis=1):
    # mean = tf.cast(mean, dtype=tf.float64)
    # logvar = tf.cast(logvar, dtype=tf.float64)
    # log2pi = tf.cast(tf.math.log(2. * np.pi), dtype=tf.float64)
    log2pi = torch.log(2. * np.pi)
    return (torch.sum(-.5 * ((sample - mean) ** 2. * torch.exp(-logvar) + logvar + log2pi)), raxis).unsqueeze(-1)
