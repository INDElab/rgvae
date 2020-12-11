"""
The function we are going to use to train the models.
In fact, this could also go in the utils file. Unsure what is the best practice.
"""
import time
import torch
import numpy as np
from tqdm import tqdm


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
        loss_val = list()
    else:
        model.train()
    loss_bar = tqdm(total=0, position=0, bar_format='{desc}')
    sanity_bar = tqdm(total=0, position=1, bar_format='{desc}')
    for target in tqdm(ds_set, total=len(ds_set), desc='Epoch {}. Batch'.format(epoch), position=2):

        loss = torch.mean(model.elbo(target))
        if not eval:
            loss.backward()
            optimizer.step()
        sanity = model.sanity_check()
        if eval:
            loss_val.append(loss.item())
        else:
            loss_bar.set_description_str('Loss: {:.6f}'.format(loss.item()))
            sanity_bar.set_description('Sanity check: {:.2f}% nodes, {:.2f}% edges, {:.2f}% adj syntax.'.format(*sanity))
    if eval:
        return np.mean(loss_val)

def train_sparse_batch(target, model, optimizer, epoch, eval: bool=False):
    """
    :param target: Dataset to train/eval on.
    Args:
        target: Dataset to train/eval on.
        model:  The model to train.
        optimizer: The optimizer for the model.
        epoch: The current epoch.
        eval: Option to switch between training and evaluation.
        sparse: the data is sparse and has to be converted.
    """
    start1 = time.time()
    loss = torch.mean(model.elbo(target))

    if not eval:
        loss.backward()
        optimizer.step()

    # sanity = model.sanity_check()

    # This is the percentage of permuted predictions.
    x_permute = 1 - torch.mean(torch.diagonal(model.x_permute, dim1=1, dim2=2)).item()
    if eval:
        return loss.item(), x_permute
    else:
        return loss.item(), x_permute
