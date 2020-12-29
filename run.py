import numpy as np
from torch_rgvae.GVAE import GVAE
from torch_rgvae.GCVAE import GCVAE
from torch_rgvae.GCVAE2 import GCVAE2
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
import wandb
import os


if __name__ == "__main__":

    # Arg parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs=1,
                        help="YAML file with configurations",
                        dest="configs",
                        type=str,
                        default=['configs/config_file.yml'])
    parser.add_argument("--dev",
                        dest="dev",
                        help="Run in develop mode",
                        nargs=1,
                        default=[1], type=int)
    arguments = parser.parse_args()

    with open(arguments.configs[0], 'r') as file:
        args = yaml.full_load(file)

    if arguments.dev[0] == 1:
        develope = True
        limit = 30
    else:
        develope = False
        limit = -1

    wandb.login(key='6d802b44b97d25931bacec09c5f1095e6c28fe36')
    print('Dev mode: {}'.format(develope))

    # Torch settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device == 'cuda':
        torch.cuda.set_device(0)
    my_dtype = torch.float64
    torch.set_default_dtype(my_dtype)

    # model_name = 'VEmbed'
    model_name = args['model_name']
    dataset = args['dataset_name']

    n = args['n']       # number of triples per matrix ( =  matrix_n/2)
    batch_size = 2**args['batch_size_exp2']        # Choose an apropiate batch size. cpu: 2**9
    if dataset == 'wn18rr' and batch_size > 2**10:                  # Avoid out of memory errors on LISA
        batch_size = 2**10
        args['batch_size_exp2'] = 10

    args['seed'] = seed = np.random.randint(1,21)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    if develope:
        # wandb.init(project="dev-mode")
        wandb.init(project="offline-dev", mode='offline')
    else:
        if 'project' in args:
            wandb.init(project=args['project'], config=args)
        else:            
            wandb.init(config=args)


    # Get data
    (n2i, i2n), (r2i, i2r), train_set, test_set, all_triples = load_link_prediction_data(dataset, use_test_set=False)
    n_e = len(n2i)
    n_r = len(r2i)
    args['n_e'] = n_e
    args['n_r'] = n_r
    truedict = truedicts(all_triples)
    dataset_tools = [truedict, i2n, i2r]

    todate = date.today().strftime("%Y%m%d")
    exp_name = args['exp_name']
    print('Experiment on the {}: {}'.format(todate, exp_name))
    print(args)

    result_dir = 'results/{}_{}'.format(exp_name, todate)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # Initialize model and optimizer.
    if model_name == 'GCVAE':
        model = GCVAE(args, n_r, n_e, dataset,).to(device)
    elif model_name == 'GCVAE2':
        model = GCVAE2(args, n_r, n_e, dataset).to(device)
    elif model_name == 'GVAE':
        model = GVAE(args, n_r, n_e, dataset).to(device)
    else:
        raise ValueError('{} not defined!'.format(model_name))

    optimizer = Ranger(model.parameters(),lr=args['lr'], k=args['k'] if 'k' in args else 9, betas=(.95,0.999), use_gc=True, gc_conv_only=False)
    wandb.watch(model)


    # Load model
    if args['load_model']:
        # model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])
        model.load_state_dict(torch.load(args['load_model_path'], map_location=torch.device(device))['model_state_dict'])
        print('Saved model loaded.')

    # Train model
    if args['train']:
        train_eval_vae(batch_size, args['epochs'], train_set[:limit], test_set[:limit], model, optimizer, dataset_tools, result_dir)
    
    # Link prediction
    if args['link_prediction']:
        print('Start link prediction!')
        testsub = torch.tensor(test_set[:300], device=d())      # TODO remove the testset croping

        lp_results =  link_prediction(model, testsub, truedict, batch_size)
        wandb.log(lp_results)
        lp_file_path = result_dir + '/lp_{}_{}.json'.format(exp_name, todate)
        with open(lp_file_path, 'w') as outfile:
            json.dump(lp_results, outfile)
        print('Saved link prediction results!')
