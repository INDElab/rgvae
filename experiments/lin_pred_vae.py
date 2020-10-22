import time
import pickle
import os
from torch_rgvae.GVAE import TorchGVAE
from torch_rgvae.GCVAE import GCVAE
from torch_rgvae.losses import *
from torch_rgvae.train_fn import train_sparse_batch
from utils import check_adj_logic
import torch
import numpy as np
from lp_utils import *
from tqdm import tqdm


# This sets the default torch dtype. Double-power
my_dtype = torch.float64
torch.set_default_dtype(my_dtype)

# Parameters. Arg parsing on its way.
n = 1   # number of triples per matrix ( =  matarix_n/2)
batch_size = 16        # Choose a low batch size for debugging, or creating the dataset will take very long.

seed = 11
np.random.seed(seed=seed)
torch.manual_seed(seed)
epochs = 111
lr = 1e-5

def train_eval_vae(n, batch_size, lr, epochs):

    # Get data
    (n2i, i2n), (r2i, i2r), train_set, test_set, all_triples = load_link_prediction_data('fb15k')
    d_n = len(n2i)
    d_e = len(r2i)

    # Initialize model and optimizer.
    model = GCVAE(n*2, d_e, d_n)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Start training.
    for epoch in range(epochs):
        start_time = time.time()
        print('Start training epoch {}'.format(epoch))
        with torch.autograd.detect_anomaly():

            model.train()
            loss_bar = tqdm(total=0, position=0, bar_format='{desc}')
            sanity_bar = tqdm(total=0, position=1, bar_format='{desc}')

            for ii in tqdm(range(len(np.ceil(train_set)/(batch_size*n))), total=len(train_set), desc='Epoch {}'.format(epoch), position=2):
                target = batch_t2m(ii, train_set, batch_size, n, d_n, d_e)
                loss, sanity = train_sparse_batch(target, model, optimizer, epoch)
                loss_bar.set_description_str('Loss: {:.6f}'.format(loss))
                sanity_bar.set_description('Sanity check: {:.2f}% nodes, {:.2f}% edges, {:.2f}% adj syntax.'.format(*sanity))

        end_time = time.time()
        print('Time elapsed for epoch{} : {:.3f}'.format(epoch, end_time - start_time))
        # Evaluate
        print("Start evaluation epoch {}.".format(epoch))
        
        with torch.no_grad():
            model.eval()
            loss_val = list()
            for ii in tqdm(range(len(np.ceil(ds_set)/(batch_size*n))), total=len(test_set), desc='Epoch {}'.format(epoch), position=2):
                target = batch_t2m(ii, train_set, batch_size, n, d_n, d_e)
                loss = train_sparse_batch(target, model, optimizer, epoch, eval=True)
                loss_val.append(loss)
        print('Epoch: {}, Test set ELBO: {:.3f}'.format(epoch, np.mean(loss_val)))

if __name__ == "__main__":

    train_eval_vae(n, batch_size, lr, epochs)
