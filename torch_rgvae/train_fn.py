"""
The function we are going to use to train the models.
In fact, this could also go in the utils file. Unsure what is the best practice.
"""
import time
import torch
from tqdm import tqdm
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
    loss_bar = tqdm(total=0, position=0, bar_format='{desc}')
    sanity_bar = tqdm(total=0, position=1, bar_format='{desc}')
    for target in tqdm(ds_set, total=len(ds_set), desc='Epoch {}. Batch'.format(epoch), position=2):
        mean, logvar = model.encode(target)
        z = model.reparameterize(mean, logvar)
        prediction = model.decode(z)

        # TODO: make it a function so we can use the same at eval.                             
        log_px_z = mpgm_loss(target, prediction)
        kl_div = kl_divergence(mean, logvar)
        loss = torch.mean( - log_px_z + kl_div)
        if not eval:
            loss.backward()
            optimizer.step()
        sanity = model.sanity_check()
        if eval:
            pass
        loss_bar.set_description_str('Loss: {:.6f}'.format(loss.item()))
        sanity_bar.set_description('Sanity check: {:.2f}% nodes, {:.2f}% edges, {:.2f}% adj syntax.'.format(*sanity))
