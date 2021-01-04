"""
Experiment: Interpolate between two subgraphs/sets of triples and print the result.
"""
import torch
import numpy as np
from torch_rgvae.GVAE import GVAE
from torch_rgvae.GCVAE import GCVAE
from torch_rgvae.train_fn import train_sparse_batch
from lp_utils import *
import pickle as pkl


def interpolate_triples(i2n, i2r, steps: int=10, model=None, model_path: str=None, i_type: str='confidence95', i_dims: tuple=(0,1,2,3,4,5,6,7,8,9)):

    if model.dataset_name == 'fb15k':
        with open('data/fb15k/e2t_dict.pkl', 'rb') as f:
            entity_dict = pkl.load(f)
    else:
        entity_dict = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    if model is None:
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = checkpoint['model_params']
        if args['model_params']['model_name'] == 'GVAE':
            model = GVAE
        elif args['model_params']['model_name'] == 'GCVAE':
            model = GCVAE
        else:
            print("Model name not found")
            raise Exception
        model.__init__(args, args['model_params']['n']*2, args['model_params']['n_r'], args['model_params']['n_e'], args['dataset_params']['dataset_name'], h_dim=args['model_params']['h_dim'], z_dim=args['model_params']['z_dim']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Model loaded.')

    z1 = torch.randn((1,model.z_dim), device=d())
    z2 = torch.randn((1,model.z_dim), device=d())

    pred_list = list()
    triples = list()
    interpolations = dict()
    # interpolations['z1'] = z1
    interpolations['between2'] = dict()
    interpolations['confidence95'] = {'confi': dict(), 'text': dict()}

    # Interpolate between z1 and z2
    if i_type == 'between2':
        print('Interpolation experiment: ' + i_type)
        step = (z2 - z1) / (steps-1)
        for i in range(steps):
            prediction = model.sample(z1 + step*i)
            prediction_json = prediction[0].detach().cpu().numpy().tolist()
            print(prediction_json)
            pred_dense = matrix2triple(prediction)
            if len(pred_dense) > 0:
                pred_list.append(prediction_json)
                text_triple = translate_triple(pred_dense, i2n, i2r, entity_dict)
                triples.append(text_triple)
                print(text_triple)
            else:
                triples.append([])
        interpolations['between2']['confi'] = pred_list
        interpolations['between2']['text'] = triples
        pred_list = list()
        triples = list()

    # Interpolating the latent space on the specified dimensions. Assuming a latent standard normal distribution.
    if i_type == 'confidence95':
        print('Interpolation experiment: ' + i_type)
        step = (1.96 * 2) / steps
        z_pred = z1.clone().detach()
        for i_dim in i_dims:
            if i_dim < model.z_dim:
                pred_list = list()
                triples = list()
                for i in range(steps):
                    z_pred[:, i_dim] = 1.96 + step * i
                    prediction = model.sample(z_pred)
                    prediction_json = prediction[0].detach().cpu().numpy().tolist()
                    print(prediction_json)
                    pred_dense = matrix2triple(prediction)
                    if len(pred_dense) > 0:
                        pred_list.append(prediction_json)
                        text_triple = translate_triple(pred_dense, i2n, i2r, entity_dict)
                        triples.append(text_triple)
                        print(text_triple)
                    else:
                        triples.append([])
                interpolations['confidence95']['confi'] = {i_dim: pred_list}
                interpolations['confidence95']['text'] = {i_dim: triples}       
    return interpolations


if __name__ == "__main__":
    pass
