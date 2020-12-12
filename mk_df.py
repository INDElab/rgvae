import pandas as pd
import pickle as pkl
import torch
import numpy as np


# This sets the default torch dtype. Double-power
my_dtype = torch.float64
torch.set_default_dtype(my_dtype)

seed = 11
np.random.seed(seed=seed)
torch.manual_seed(seed)
n = 1       # Number of triples per graph
steps = 10   # Interpolation steps

dataset = 'fb15k'
model_path = '/home/fwolf/results/tune_GCVAE_fb15k_b100_20201206' + '/rgvae_dict.pt'

# Load the entity dictionary
entity2text = pd.read_csv('data/fb15k/entity2text.txt', header=None, sep='\t')
entity2text.columns = ['Entity', 'Text']
entity_dict = entity2text.set_index('Entity').T.to_dict('series')
del entity2text

file_path = 'data/fb15k/e2t_dict.pkl'
with open(file_path, 'wb') as f:
    pkl.dump(entity_dict, f)
