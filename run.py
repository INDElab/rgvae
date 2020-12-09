import numpy as np
from torch_rgvae.GVAE import GVAE
from torch_rgvae.GCVAE import GCVAE
from torch_rgvae.VEmbed import VEmbed 
from lp_utils import *
from experiments.train_eval_vae import train_eval_vae
from experiments.link_prediction import link_prediction
from experiments.lp_vembed import train_lp_vembed
from datetime import date
import yaml, json
import argparse
import torch
import torch_optimizer as optim
from ranger import Ranger
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--configs', nargs=1,
    #                     help="YAML file with configurations",
    #                     type=argparse.FileType('r'),
    #                     default='configs/config_file.yml')
    # arguments = parser.parse_args()
    # args = yaml.full_load(arguments.configs[0])

    with open('configs/config_file.yml', 'r') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)

    # model_name = args['model_params']['model_name']
    model_name = 'VEmbed'
    n = args['model_params']['n']       # number of triples per matrix ( =  matrix_n/2)
    batch_size = 2**args['model_params']['batch_size_exp2']        # Choose an apropiate batch size. cpu: 2**9
    h_dim = args['model_params']['h_dim']       # number of hidden dimensions
    z_dim = args['model_params']['z_dim']      # number of latent dimensions
    beta = args['model_params']['beta']         # beta parameter of betaVAE
    seed = 11
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    epochs = args['model_params']['epochs']
    lr = args['model_params']['lr']
    k = args['model_params']['k'] if 'k' in args['model_params'] else 6

    dataset = args['dataset_params']['dataset_name']
    model_path = args['experiment']['load_model_path']


    # Get data
    (n2i, i2n), (r2i, i2r), train_set, test_set, all_triples = load_link_prediction_data(dataset, use_test_set=False)
    d_n = len(n2i)
    d_e = len(r2i)

    todate = date.today().strftime("%Y%m%d")
    exp_name = args['experiment']['exp_name']
    print('Experiment: ' + exp_name)
    
    result_dir = 'results/{}_{}'.format(exp_name, todate)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # Initialize model and optimizer.
    if model_name == 'GCVAE':
        model = GCVAE(n*2, d_e, d_n, dataset, h_dim=h_dim, z_dim=z_dim, beta=beta).to(device)
    elif model_name == 'GVAE':
        model = GVAE(n*2, d_e, d_n, dataset, h_dim=h_dim, z_dim=z_dim, beta=beta).to(device)
    elif model_name == 'VEmbed':
        model = VEmbed(d_n, d_e, z_dim=z_dim)
    else:
        raise ValueError('{} not defined!'.format(model_name))

    # optimizer = optim.Ranger(model.parameters(),lr=lr, k=11, betas=(.95,0.999), use_gc=True, gc_conv_only=False, )
    optimizer = Ranger(model.parameters(),lr=lr, k=k, betas=(.95,0.999), use_gc=True, gc_conv_only=False)

    # Load model
    if args['experiment']['load_model']:
        # model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])
        print('Saved model loaded.')

    # Train model
    if args['experiment']['train']:
        if model_name == "VEmbed":
            train_lp_vembed(model, optimizer, train_set[:3], test_set[:9], all_triples, epochs, batch_size, result_dir)
        else:
            train_eval_vae(n, batch_size, epochs, train_set, test_set, model, optimizer, result_dir)

    # Link prediction
    if args['experiment']['link_prediction']:
        if model_name == 'VEmbed':
            pass
        else:
            print('Start link prediction!')
            testsub = torch.tensor(test_set[:300], device=d())      # TODO remove the testset croping
            truedict = truedicts(all_triples)

            lp_results =  link_prediction(model, testsub, truedict, batch_size)
            
            lp_file_path = result_dir + '/lp_{}_{}.json'.format(exp_name, todate)
            with open(lp_file_path, 'w') as outfile:
                json.dump(lp_results, outfile)
            print('Saved link prediction results!')
