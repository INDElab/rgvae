import torch
from torch_rgvae.RGVAE import TorchRGVAE
from utils.lp_utils import *


args = {}
# Get data
final = args['final'] if 'final' in args else False
(n2i, i2n), (r2i, i2r), train_set, test_set, all_triples = load_link_prediction_data('fb15k', use_test_set=final)
n_e = len(n2i)
n_r = len(r2i)
args['n_e'] = n_e
args['n_r'] = n_r
args['n'] = n = 3
truedict = truedicts(all_triples)
dataset_tools = [truedict, i2n, i2r]
# Train with part data

model = TorchRGVAE(args, n_r, n_e, train_set[:100], 'fb15k')
batch = train_set[11:11+n]
print(batch)
pred = model(batch)

print(pred.numpy())