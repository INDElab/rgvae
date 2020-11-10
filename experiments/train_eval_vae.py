import time
import os
from torch_rgvae.GVAE import GVAE
from torch_rgvae.GCVAE import GCVAE
from torch_rgvae.train_fn import train_sparse_batch
import torch
import numpy as np
from lp_utils import *
from tqdm import tqdm
from datetime import date


def train_eval_vae(n, batch_size, epochs, train_set, test_set, model, optimizer):

    d_n = model.na
    d_e = model.ea

    old_loss = 3333
    loss_dict = {}

    # Start training.
    for epoch in range(epochs):
        start_time = time.time()
        print('Start training epoch {}'.format(epoch))
        torch.backends.cudnn.benchmark = True
        model.train()
        loss_bar = tqdm(total=0, position=0, bar_format='{desc}')
        sanity_bar = tqdm(total=0, position=1, bar_format='{desc}')
        for b_from in tqdm(range(0,len(train_set),(batch_size*n)), desc='Epoch {}'.format(epoch), position=2):
            b_to = min(b_from + batch_size, len(train_set))
            target = batch_t2m(torch.tensor(train_set[b_from:b_to], device=d()), n, d_n, d_e)

            loss, sanity, x_permute = train_sparse_batch(target, model, optimizer, epoch)

            loss_bar.set_description_str('Loss: {:.6f}'.format(loss))
            sanity_bar.set_description('Sanity check: {:.2f}% nodes, {:.2f}% edges, {:.2f}% permuted.'.format(*sanity,x_permute*100))
                
        end_time = time.time()
        print('Time elapsed for epoch{} : {:.3f}'.format(epoch, end_time - start_time))


        # Evaluate
        print("Start evaluation epoch {}.".format(epoch))
        with torch.no_grad():
            model.eval()
            loss_val = list()
            permute_list = list()
            for b_from in tqdm(range(0,len(test_set),(batch_size*n)), desc='Epoch {}'.format(epoch), position=2):
                b_to = min(b_from + batch_size, len(test_set))
                target = batch_t2m(torch.tensor(test_set[b_from:b_to], device=d()), n, d_n, d_e)
                loss, x_permute = train_sparse_batch(target, model, optimizer, epoch, eval=True)
                loss_val.append(loss)
                permute_list.append(x_permute)
        mean_loss = np.mean(loss_val)
        loss_dict[epoch] = mean_loss
        print('Epoch: {}, Test set ELBO: {:.3f}, permuted {:.3f}%'.format(epoch, mean_loss, np.mean(permute_list)*100))

        if 'old_loss' in locals() and mean_loss < old_loss and epoch > 6:
            # Check for data folder and eventually create.
            if not os.path.isdir('data/model'):
                os.mkdir('data/model')
            torch.save(model.state_dict(), 'data/model/{}_{}_{}e_{}l_{}.pt'.format(model.name, model.dataset_name, epoch, int(mean_loss), date.today().strftime("%Y%m%d")))
            old_loss = mean_loss
            print('Model saved at epoch {}'.format(epoch))
    return loss_dict
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
    epochs = 111
    lr = 1e-5
    dataset = 'wn18rr'
    # model_path = 'data/model/GCVAE_fb15k_69e_20201025.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Get data
    (n2i, i2n), (r2i, i2r), train_set, test_set, all_triples = load_link_prediction_data(dataset)
    d_n = len(n2i)
    d_e = len(r2i)

    # Initialize model and optimizer.
    model = GCVAE(n*2, d_e, d_n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    print('This should not be printed.')

    loss_dict = train_eval_vae(n, batch_size, lr, epochs, dataset)
