import time
import pickle
import os
from torch_rgvae.GVAE import TorchGVAE
from torch_rgvae.GCVAE import GCVAE
from torch_rgvae.losses import *
from torch_rgvae.train_fn import train_epoch
from utils.utils import check_adj_logic
import torch
import numpy as np

"""
Holds the training and eval functions for the different models.
For now it executes the training and evaluation of the RGVAE with radom data.
Change the parameters in-script. Parsing is yet to come.
"""


# This sets the default torch dtype. Double-power
my_dtype = torch.float64
torch.set_default_dtype(my_dtype)

# Parameters. Arg parsing on its way.
n = 5
e = 10      # All random graphs shall have 5 nodes and 10 edges
d_e = 11
d_n = 55
batch_size = 16        # Choose a low batch size for debugging, or creating the dataset will take very long.
params = [n, e, d_e, d_n, batch_size]

seed = 11
np.random.seed(seed=seed)
torch.manual_seed(seed)
epochs = 111
lr = 1e-5

# Initialize model and optimizer.
model = GCVAE(n, d_e, d_n)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)



def train_eval_vae(params, model, optimizer, epochs):
    n, e, d_e, d_n, batch_size = params
    data_file = 'data/graph_ds_n{}_e{}_de{}_dn{}_bs{}.pkl'.format(n, e, d_e, d_n, batch_size)

    # Check for data folder and eventually create.
    if not os.path.isdir('data'):
        os.mkdir('data')

    # Check for data set and eventually create.
    if os.path.isfile(data_file):
        with open(data_file, "rb") as fp:
            print('Loading dataset..')
            train_set, test_set = pickle.load(fp)
    else:
        print('Creating dataset..')
        train_set = mk_graph_ds(n, d_e, d_n, e, batches=1000, batch_size=batch_size)
        test_set = mk_graph_ds(n, d_e, d_n, e, batches=200, batch_size=batch_size)
        with open(data_file, "wb") as fp:
            pickle.dump([train_set, test_set], fp)

    # Start training.
    for epoch in range(epochs):
        start_time = time.time()
        print('Start training epoch {}'.format(epoch))
        with torch.autograd.detect_anomaly():
            train_epoch(train_set, model, optimizer, epoch)
        end_time = time.time()
        print('Time elapsed for epoch{} : {:.3f}'.format(epoch, end_time - start_time))
        # Evaluate
        print("Start evaluation epoch {}.".format(epoch))
        with torch.no_grad():
            mean_loss = train_epoch(test_set, model, optimizer, epoch, eval=True)
            print('Epoch: {}, Test set ELBO: {:.3f}, time elapse for current epoch: {:.3f}'.format(epoch, mean_loss))

if __name__ == "__main__":

    train_eval_vae(params, model, optimizer, epochs)
