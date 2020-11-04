"""
Experiment: Interpolate between two subgraphs/sets of triples and print the result.
"""
import torch
import numpy as np
from torch_rgvae.GVAE import TorchGVAE
from torch_rgvae.GCVAE import GCVAE
from torch_rgvae.train_fn import train_sparse_batch
from lp_utils import *


# This sets the default torch dtype. Double-power
my_dtype = torch.float64
torch.set_default_dtype(my_dtype)

seed = 11
np.random.seed(seed=seed)
torch.manual_seed(seed)
n = 1       # Number of triples per graph
steps = 5   # Interpolation steps

dataset = 'fb15k'
model_path = 'data/model/GCVAE_fb15k_85e_20201025.pt'


def interpolate_triples(n: int, steps: int, data_set: str, model_path: str):
     # Get data
    (n2i, i2n), (r2i, i2r), train_set, test_set, all_triples = load_link_prediction_data(dataset)
    d_n = len(n2i)
    d_e = len(r2i)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = GCVAE(n*2, d_e, d_n).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    print('Model loaded.')

    rand1, rand2 = np.random.randint(0, len(test_set), size=2)

    # Encode:
    target1 = triple2matrix(test_set[rand1:rand1+n], d_n, d_e)
    mean, logvar = model.encode(target1)
    z1 = model.reparameterize(mean, logvar)

    target2 = triple2matrix(test_set[rand2:rand2+n], d_n, d_e)
    mean, logvar = model.encode(target2)
    z2 = model.reparameterize(mean, logvar)
    prediction2 = model.sample(z1)

    # Interpolate between z1 and z2
    step = (z2 - z1) / steps
    pred_list = list()
    triples = list()

    for i in range(steps):
        prediction = model.sample(z1 + step*i)
        pred_list.append(prediction)
        pred_dense = matrix2triple(prediction)
        if len(pred_dense) > 0:
            triples.append(pred_dense)
            print(translate_triple(triples[-1], i2n, i2r))

    pred_list.append(prediction2)
    triples.append(matrix2triple(prediction2))



if __name__ == "__main__":
    interpolate_triples(n, steps, dataset, model_path)
