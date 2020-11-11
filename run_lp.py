
import numpy as np
from torch_rgvae.GVAE import GVAE
from torch_rgvae.GCVAE import GCVAE
from lp_utils import *
from experiments.train_eval_vae import train_eval_vae
from experiments.link_prediction import link_prediction
from datetime import date
import json
import argparse
import torch
import os


if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', nargs=1,
    #                     help="JSON file with configurations",
    #                     type=argparse.FileType('r'))
    # arguments = parser.parse_args()
    # # Loading a JSON object returns a dict.
    # config = json.load(arguments.config)
    # print(arguments)

    # This sets the default torch dtype. Double-power
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device == 'cuda':
        torch.cuda.set_device(0)
    my_dtype = torch.float64
    torch.set_default_dtype(my_dtype)

    # Parameters. Arg parsing on its way.
    n = 1       # number of triples per matrix ( =  matrix_n/2)
    batch_size = 2**13        # Choose a low batch size for debugging,.
    h = 60      # number of hidden dimensions
    seed = 11
    np.random.seed(seed=seed)
    torch.manual_seed(seed)


    for dataset in ['fb15k', 'wn18rr']:
        model_list = list()
        folder = 'data/{}/models/'.format(dataset)
        for filename in os.listdir(folder):
            if filename.endswith(".pt"):
                model_list.append(filename)
        # Get data
        (n2i, i2n), (r2i, i2r), train_set, test_set, all_triples = load_link_prediction_data(dataset, use_test_set=False)
        d_n = len(n2i)
        d_e = len(r2i)
        for model_path in model_list:
            # Initialize model and optimizer.
            if 'GCVAE' in model_path.split('_'):
                model = GCVAE(n*2, d_e, d_n, dataset, z_dim=h).to(device)
            else:
                model = GVAE(n*2, d_e, d_n, dataset, z_dim=h).to(device)


            model.load_state_dict(torch.load(folder + model_path, map_location=torch.device(device)))
            print('Saved model loaded.')
            
            testsub = torch.tensor(test_set[:2], device=d())
            truedict = truedicts(all_triples)

            lp_results =  link_prediction(model, testsub, truedict, batch_size)

            lp_file_path = 'data/'+dataset+'/lp_results_{}_{}.json'.format(model.name, date.today().strftime("%Y%m%d"))
            with open(lp_file_path, 'w') as outfile:
                json.dump(lp_results, outfile)
