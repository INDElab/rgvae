"""
The function we are going to use to train the models.
In fact, this could also go in the utils file. Unsure what is the best practice.
"""
import torch
from torch_rgvae.losses import *


def train_epoch(ds_set, model, optimizer, epoch, eval: bool=False):
    """
    Args:
        ds_set: Dataset to train/eval on.
        model:  The model to train.
        optimizer: The optimizer for the model.
        epoch: The current epoch.
        eval: Option to switch between training and evaluation.
    """
    if eval:
        model.eval()
    else:
        model.train()
    for target in ds_set:
        mean, logvar = model.encode(target)
        z = model.reparameterize(mean, logvar)
        prediction = model.decode(z)

        # TODO: make it a function so we can use the same at eval.                             
        log_px_z = mpgm_loss(target, prediction)
        kl_div = kl_divergence(mean, logvar)
        loss = torch.mean( - log_px_z + kl_div)
        print('Epoch {} \n loss {}'.format(epoch, loss.item()))
        if not eval:
            loss.backward()
            optimizer.step()
        sanity = model.sanity_check()
        print('Sanity check: {:.2f}% nodes, {:.2f}% edges, {:.2f}% adj syntax.'.format(*sanity))
