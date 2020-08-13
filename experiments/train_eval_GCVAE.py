import time
import pickle
import os
from torch_rgvae.RGVAE import TorchRGVAE
from torch_rgvae.losses import *
import torch
import numpy as np
from utils import *

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
d_e = 3
d_n = 3
batch_size = 64
params = [n, e, d_e, d_n, batch_size]

seed = 11
np.random.seed(seed=seed)
torch.manual_seed(seed)
epochs = 111
lr = 1e-5


def train_eval_RGCVAE(params, epochs, lr=1e-5):
    n, e, d_e, d_n, batch_size = params
    data_file = 'data/Sparse_graph_ds_n{}_e{}_de{}_dn{}.pkl'.format(n, e, d_e, d_n)

    # Check for data folder and eventually create.
    if not os.path.isdir('data'):
        os.mkdir('data')

    # Check for data set and eventually create.
    if os.path.isfile(data_file):
        with open(data_file, "rb") as fp:
            train_set, test_set = pickle.load(fp)
    else:
        train_set = mk_sparse_graph_ds(n, e, d_e, batches=400, batch_size=batch_size)
        test_set = mk_sparse_graph_ds(n, e, d_e, batches=100, batch_size=batch_size)
        with open(data_file, "wb") as fp:
            pickle.dump([train_set, test_set], fp)

    # Initialize model and optimizer.

    ###############################################
    ###         TODO: Init with triples        ###
    ###############################################

    model = TorchRGVAE(n, d_e, d_n)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Start training.
    for epoch in range(epochs):
        start_time = time.time()
        print('Start training epoch {}'.format(epoch))
        with torch.autograd.detect_anomaly():
            for target in train_set:
                model.train()
                mean, logstd = model.encode(target)
                z = model.reparameterize(mean, logstd)
                prediction = model.decode(z)

                # My personal collection of losses. Mix and match as you like :)
                # TODO: make it a function so we can use the same at eval.
                log_pz = log_normal_pdf(z, torch.zeros_like(z), torch.zeros_like(z))
                log_qz_x = log_normal_pdf(z, mean, 2 * logstd)
                log_px = mpgm_loss(target, prediction)
                bce_loss = graph_loss(target, prediction)
                G_loss = torch.mean(log_px + log_pz + log_qz_x)
                G_std_loss = std_loss(prediction)
                loss = bce_loss + G_loss + G_std_loss
                print('Epoch {} \n loss {}'.format(epoch, loss.item()))
                loss.backward()
                optimizer.step()
                end_time = time.time()

        # Evaluate
        print("Start evaluation epoch {}.".format(epoch))
        mean_loss = []
        with torch.no_grad():
            model.eval()
            for test_x in test_set:
                mean, logstd = model.encode(target)
                z = model.reparameterize(mean, logstd)
                prediction = model.decode(z)
                log_pz = log_normal_pdf(z, torch.zeros_like(z), torch.zeros_like(z))
                log_qz_x = log_normal_pdf(z, mean, 2 * logstd)
                log_px = mpgm_loss(target, prediction)
                loss = - torch.mean(log_px + log_pz + log_qz_x)
                mean_loss.append(loss)
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, np.mean(mean_loss),
                                                                                           end_time - start_time))

if __name__ == "__main__":

    train_eval_RGCVAE(params, epochs, lr)