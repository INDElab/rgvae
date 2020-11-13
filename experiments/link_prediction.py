
import time
import os
from torch_rgvae.GVAE import GVAE
from torch_rgvae.GCVAE import GCVAE
from torch_rgvae.train_fn import train_sparse_batch
from lp_utils import *
import tqdm
import json
from datetime import date


def link_prediction(model, testsub, truedict, batch_size):
    """
    Performs linkpredcition with the given model on the gives data's testset.
    Saves results as json in /data folder.
    :param model: torch VAE model
    :param dataset: name or the dataset
    :param truedict: collection of true tripples per head+rel/tail+rel set
    :param batch_size: batch size
    """

    d_n = model.na
    d_e = model.ea

    with torch.no_grad():

        model.train(False)

        mrr, hits, ranks = eval(
            model=model, valset=testsub, truedicts=truedict, n=d_n, r=d_e,
            batch_size=batch_size, verbose=True, elbo=True)

    print(f'MRR {mrr:.4}\t hits@1 {hits[0]:.4}\t  hits@3 {hits[1]:.4}\t  hits@10 {hits[2]:.4}')

    lp_results = {}
    lp_results[model.name] = []
    lp_results[model.name].append({'mrr': mrr,
                                'h@1': hits[0],
                                'h@3': hits[1],
                                'h@10': hits[2]})
    return lp_results



if __name__ == "__main__":

    # This sets the default torch dtype. Double-power
    my_dtype = torch.float64
    torch.set_default_dtype(my_dtype)

    # Parameters. Arg parsing on its way.
    n = 1       # number of triples per matrix ( =  matrix_n/2)
    batch_size = 16        # Choose a low batch size for debugging, or creating the dataset will take very long.

    seed = 11
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    dataset = 'fb15k'
    model_path = 'data/model/GCVAE_fb15k_11e_20201025.pt'
                                                
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Get data
    (n2i, i2n), (r2i, i2r), train_set, test_set, all_triples = load_link_prediction_data(dataset)
    d_n = len(n2i)
    d_e = len(r2i)

    testsub = torch.tensor(test_set)
    truedict = truedicts(all_triples)

    # Initialize model.
    model = GCVAE(n*2, d_e, d_n).to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        print('Saved model loaded.')

    lp_results =  load_link_prediction(model, test_set, truedict, d_n, d_r, batch_size)

    outfile_path = 'data/'+dataset+'/lp_results_{}_{}.txt'.format(model.name, date.today().strftime("%Y%m%d"))
    with open(outfile_path, 'w') as outfile:
        json.dump(lp_results, outfile)
