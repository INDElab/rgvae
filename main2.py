"""
Main script v2 for VAE with Kipft RGCN. We concat the F and E matrix and make A a spare matrix.
For now it executes the training and evaluation of the RGVAE with radom data.
Change the parameters in-script. Parsing is yet to come.
"""
import time
import torch
import numpy as np
from models.GVAE import TorchGVAE
from models.torch_losses import *
from models.utils import *


# This sets the default torch dtype. Double-power
my_dtype = torch.float64
torch.set_default_dtype(my_dtype)

# Parameters
n = 5
d_e = 3
d_n = 2

seed = 11
np.random.seed(seed=seed)
torch.manual_seed(seed)
epochs = 111
batch_size = 64

train_set = mk_random_graph_ds(n, d_e, d_n, 4000, batch_size=batch_size)
test_set = mk_random_graph_ds(n, d_e, d_n, 1000, batch_size=batch_size)

model = TorchGVAE(n, d_e, d_n)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=5e-4)

for epoch in range(epochs):
    start_time = time.time()

    for target in train_set:
        model.train()
        mean, logstd = model.encode(target)
        z = model.reparameterize(mean, logstd)
        prediction = model.decode(z)

        log_pz = log_normal_pdf(z, torch.zeros_like(z), torch.zeros_like(z))
        log_qz_x = log_normal_pdf(z, mean, 2*logstd)
        log_px = mpgm_loss(target, prediction)
        loss = - torch.mean(log_px + log_pz + log_qz_x)
        print(loss)
        loss.backward()
        optimizer.step()
        end_time = time.time()

    # Evaluate
    mean_loss = []
    with torch.no_grad():
        model.eval()
        for test_x in test_set:
            mean, logstd = model.encode(target)
            z = model.reparameterize(mean, logstd)
            prediction = model.decode(z)
            log_pz = log_normal_pdf(z, torch.zeros_like(z), torch.zeros_like(z))
            log_qz_x = log_normal_pdf(z, mean, 2*logstd)
            log_px = mpgm_loss(target, prediction)
            loss = - torch.mean(log_px + log_pz + log_qz_x)
            mean_loss.append(loss)
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, np.mean(mean_loss), end_time - start_time))

