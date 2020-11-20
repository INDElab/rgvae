import numpy as np
from torch_rgvae.GVAE import GVAE
from torch_rgvae.GCVAE import GCVAE
from lp_utils import *
from experiments.train_eval_vae import train_eval_vae
from experiments.link_prediction import link_prediction
from datetime import date
import yaml, json
import argparse
import torch
import os


if __name__ == "__main__":
    

    # Torch settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device == 'cuda':
        torch.cuda.set_device(0)
    my_dtype = torch.float64
    torch.set_default_dtype(my_dtype)

    # Arg parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs=1,
                        help="YAML file with configurations",
                        type=argparse.FileType('r'))
    arguments = parser.parse_args()
    args = yaml.full_load(arguments.configs[0])

    model_name = args['model_params']['model_name']
    n = args['model_params']['n']       # number of triples per matrix ( =  matrix_n/2)
    batch_size = 2**args['model_params']['batch_size_exp2']        # Choose an apropiate batch size. cpu: 2**9
    z_dim = args['model_params']['z_dim']      # number of latent dimensions
    seed = 11
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    epochs = args['model_params']['epochs']
    lr = args['model_params']['lr']

    dataset = args['dataset_params']['dataset_name']
    model_path = args['experiment']['load_model_path']

    # Get data
    (n2i, i2n), (r2i, i2r), train_set, test_set, all_triples = load_link_prediction_data(dataset, use_test_set=False)
    d_n = len(n2i)
    d_e = len(r2i)

    result_dir = 'results/exp_{}'.format(date.today().strftime("%Y%m%d"))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # Initialize model and optimizer.
    if model_name == 'GCVAE':
        model = GCVAE(n*2, d_e, d_n, dataset, z_dim=z_dim).to(device)
    elif model_name == 'GVAE':
        model = GVAE(n*2, d_e, d_n, dataset, z_dim=z_dim).to(device)
    else:
        raise ValueError('{} not defined!'.format(model_name))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Load model
    if args['experiment']['load_model']:
        # model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        print('Saved model loaded.')

    # Train model
    if args['experiment']['train']:
        train_eval_vae(n, batch_size, epochs, train_set, test_set, model, optimizer, result_dir)

    # Link prediction
    if args['experiment']['link_prediction']:
        testsub = torch.tensor(test_set, device=d())
        truedict = truedicts(all_triples)

        lp_results =  link_prediction(model, testsub, truedict, batch_size)
        

        lp_file_path = result_dir + '/lp_results_{}_{}.json'.format(model.name, dataset)
        with open(lp_file_path, 'w') as outfile:
            json.dump(lp_results, outfile)
