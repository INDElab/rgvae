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
    num_no_improvements = 0
    loss_dict = {'val': dict(), 'train': dict()}

    # Start training.
    for epoch in range(epochs):
        start_time = time.time()
        print('Start training epoch {}'.format(epoch))
        torch.backends.cudnn.benchmark = True
        model.train()
        loss_bar = tqdm(total=0, position=0, bar_format='{desc}')
        sanity_bar = tqdm(total=0, position=1, bar_format='{desc}')
        loss_train = list()

        for b_from in tqdm(range(0,len(train_set),(batch_size*n)), desc='Epoch {}'.format(epoch), position=2):
            b_to = min(b_from + batch_size, len(train_set))
            target = batch_t2m(torch.tensor(train_set[b_from:b_to], device=d()), n, d_n, d_e)

            loss, sanity, x_permute = train_sparse_batch(target, model, optimizer, epoch)
            loss_train.append(loss)
            loss_bar.set_description_str('Loss: {:.6f}'.format(loss))
            sanity_bar.set_description('Sanity check: {:.2f}% nodes, {:.2f}% edges, {:.2f}% permuted.'.format(*sanity,x_permute*100))
        
        loss_dict['train'][epoch] = np.mean(loss_train)
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
        loss_dict[val][epoch] = mean_loss
        print('Epoch: {}, Test set ELBO: {:.3f}, permuted {:.3f}%'.format(epoch, mean_loss, np.mean(permute_list)*100))

        if mean_loss < old_loss and epoch > 11:
            # Check for data folder and eventually create.
            if not os.path.isdir('data/model'):
                os.mkdir('data/model')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'loss': mean_loss
                # TODO can I save the loss dict here too?
            },  'data/model/{}_{}_{}.pt'.format(model.name, model.dataset_name, date.today().strftime("%Y%m%d")))
            num_no_improvements = 0
            old_loss = mean_loss
            print('Model saved at epoch {}'.format(epoch))
        # Early stopping
        else:
            num_no_improvements += 1
        if num_no_improvements == 6:
            print('Early Stopping at epoch {}'.format(epoch))
            break
        else:
            continue
    return loss_dict


if __name__ == "__main__":
    pass
