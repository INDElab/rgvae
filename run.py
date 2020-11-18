import numpy as np
from torch_rgvae.GVAE import GVAE
from torch_rgvae.GCVAE import GCVAE
from lp_utils import *
from experiments.train_eval_vae import train_eval_vae
from experiments.link_prediction import link_prediction
from datetime import date
import yaml
import argparse
import torch


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs=1,
                        help="YAML file with configurations",
                        type=argparse.FileType('r'))
    arguments = parser.parse_args()

    # Loading a JSON object returns a dict.
    args = yaml.full_load(arguments.configs[0])
    print(args)

    # This sets the default torch dtype. Double-power
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device == 'cuda':
        torch.cuda.set_device(0)
    my_dtype = torch.float64
    torch.set_default_dtype(my_dtype)

    # Parameters. Arg parsing on its way.
    n = args['model_params']['n']       # number of triples per matrix ( =  matrix_n/2)
    batch_size = 2**args['model_params']['batch']        # Choose a low batch size for debugging, or creating the dataset will take very long.
    z_dim = args['model_params']['z_dim']      # number of hidden dimensions
    seed = 11
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    epochs = args['model_params']['epochs']
    lr = args['model_params']['lr']
    # model_path = 'data/model/GCVAE_fb15k_69e_20201025.pt'


    for dataset in ['fb15k', 'wn18rr']:
        # Get data
        (n2i, i2n), (r2i, i2r), train_set, test_set, all_triples = load_link_prediction_data(dataset, use_test_set=False)
        d_n = len(n2i)
        d_e = len(r2i)
        for model_name in ['GCVAE']:
            # Initialize model and optimizer.
            if model_name == 'GCVAE':
                model = GCVAE(n*2, d_e, d_n, dataset, z_dim=z_dim).to(device)
            else:
                model = GVAE(n*2, d_e, d_n, dataset, z_dim=z_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

            if 'model_path' in locals():
                model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
                print('Saved model loaded.')


            loss_dict =  train_eval_vae(n, batch_size, epochs, train_set, test_set, model, optimizer)

            loss_file_path = 'data/model/{}_{}_{}.json'.format(model.name, model.dataset_name, date.today().strftime("%Y%m%d"))
            with open(loss_file_path, 'w') as outfile:
                json.dump(loss_dict, outfile)

            testsub = torch.tensor(test_set[:2], device=d())
            truedict = truedicts(all_triples)

            lp_results =  link_prediction(model, testsub, truedict, batch_size)

            lp_file_path = 'data/'+dataset+'/lp_results_{}_{}.json'.format(model.name, date.today().strftime("%Y%m%d"))
            with open(lp_file_path, 'w') as outfile:
                json.dump(lp_results, outfile)
