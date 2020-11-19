import torch
from lp_utils import *
from experiments.link_prediction import link_prediction
from torch_rgvae.GCVAE import GCVAE

# Get data
(n2i, i2n), (r2i, i2r), train_set, test_set, all_triples = load_link_prediction_data('fb15k', use_test_set=False)
d_n = len(n2i)
d_e = len(r2i)
device= 'cpu'
model = GCVAE(1*2, d_e, d_n, 'fb15k', z_dim=60).to(device)

loaded = torch.load('/home/wolf/Thesis/Code/rgvae/results/exp_20201119/GCVAE_fb15k_20201119.pt', map_location=torch.device(device))

loss_dict = loaded['loss_log']
print(loss_dict)

model.load_state_dict(loaded['model_state_dict'])
print('Saved model loaded.')