import os, json, time
from torch.utils.tensorboard import SummaryWriter
from torch_rgvae.train_fn import train_sparse_batch
from experiments.link_prediction import link_prediction
from experiments.interpolation import interpolate_triples 
import torch
import numpy as np
from lp_utils import *
from tqdm import tqdm
from datetime import date
import wandb, random



def train_eval_vae(batch_size, epochs, train_set, test_set, model, optimizer, dataset_tools, result_dir, final: bool=False):
    """
    Train and evaluate the model on the test and train set.
    :param n: triples per graph
    :param batch_size: bs
    :param epochs: laps
    :param train_set: train data
    :param test_set: test data
    :param model: Pytorch RGVAE model
    :param optimizer: mice r
    :param result_dir: path where to save model state_dict
    :returns : dict with train and val loss per epoch
    """
    n = int(model.n / 2)
    n_e = model.n_e
    n_r = model.n_r
    truedict, i2n, i2r = dataset_tools

    old_loss = best_loss = 3333.
    loss_dict = {'val': dict(), 'train': dict(), 'lp': dict()}
    writer = SummaryWriter(log_dir=result_dir)

    # Start training.
    for epoch in range(epochs):
        start_time = time.time()
        print('Start training epoch {}'.format(epoch))
        torch.backends.cudnn.benchmark = True
        model.train()
        loss_bar = tqdm(total=0, position=0, bar_format='{desc}')
        loss_train = list()
        permute_list = list()

        for b_from in tqdm(range(0,len(train_set),(batch_size*n)), desc='Epoch {}'.format(epoch), position=2):
            b_to = min(b_from + batch_size, len(train_set))
            target = batch_t2m(torch.tensor(train_set[b_from:b_to], device=d()), n, n_e, n_r)

            loss, x_permute = train_sparse_batch(target, model, optimizer, epoch)
            loss_train.append(loss)
            permute_list.append(x_permute)
            loss_bar.set_description_str('Loss: {:.6f}'.format(loss))
            writer.add_scalar('Loss/train', loss, epoch)
            wandb.log({"train_loss_step": loss})
        
        loss_dict['train'][epoch] = loss_train
        wandb.log({"train_loss_mean": np.mean(loss_train), "train_loss_std": np.std(loss_train), 
                    "train_permutation_mean": np.mean(permute_list), "train_permutation_std": np.std(permute_list), "epoch": epoch})
        end_time = time.time()
        print('Time elapsed for epoch{} : {:.3f}\n Mean train elbo: {:.3f}'.format(epoch, end_time - start_time, np.mean(loss_train)))


        # Evaluate
        if final and (epoch+1) == epochs:
            # save model last
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model_params': model.model_params,
            'loss_log': loss_dict},
            result_dir + '/rgvae_dict_final.pt')
        elif final:
            pass
        else:
            print("Start evaluation epoch {}.".format(epoch))
            testsub = torch.tensor(test_set, device=d())[random.sample(range(len(test_set)), k=50)]   # TODO remove the testset croping
            
            with torch.no_grad():
                model.eval()
                loss_val = list()
                permute_list = list()
                for b_from in tqdm(range(0,len(test_set),(batch_size*n)), desc='Epoch {}'.format(epoch), position=2):
                    b_to = min(b_from + batch_size, len(test_set))
                    target = batch_t2m(torch.tensor(test_set[b_from:b_to], device=d()), n, n_e, n_r)
                    loss, x_permute = train_sparse_batch(target, model, optimizer, epoch, eval=True)
                    loss_val.append(loss)
                    permute_list.append(x_permute)
                    writer.add_scalar('Loss/test', loss, epoch)
                    wandb.log({"val_loss_step": loss})
            mean_loss = np.mean(loss_val)
            loss_dict['val'][epoch] = loss_val
            wandb.log({"val_loss_mean": mean_loss, "val_loss_std": np.std(loss_val), 
                        "val_permutation_mean": np.mean(permute_list), "val_permutation_std": np.std(permute_list), "epoch": epoch})
            print('Epoch: {}, Mean eval elbo: {:.3f}, permuted {:.2f}%'.format(epoch, mean_loss, np.mean(permute_list)*100))

            # Do Link prediction
            # if epoch % 30 == 0:
            if (epoch+1) % 20 == 0 or (epoch+1) == epochs:
                print('Start interpolating the latent space and generating triples at epoch {}.'.format(epoch))
                interpolations = interpolate_triples(i2n,i2r, 5, model)
                wandb.log({"interpolations": interpolations, "epoch": epoch})
                interpol_file_path = result_dir + '/interpolation_e{}.json'.format(epoch)
                with open(interpol_file_path, 'w') as f:
                    json.dump(interpolations, f)
                wandb.save(interpol_file_path)

                print('Start link prediction at epoch {}:'.format(epoch))
                lp_start = time.time()
                lp_results =  link_prediction(model, testsub, truedict, batch_size)
                loss_dict['lp'][epoch] = lp_results
                wandb.log(lp_results)
                lp_end = time.time()
                lp_file_path = result_dir + '/lp_e{}.json'.format(epoch)
                with open(lp_file_path, 'w') as outfile:
                    json.dump(lp_results, outfile)
                wandb.save(lp_file_path)
                print('Saved link prediction results!')
                print('Time elapsed for Link prediction at epoch{} : {:.3f}'.format(epoch, lp_end - lp_start))
                print('MRR {:.4}\t hits@1 {:.4}\t  hits@3 {:.4}\t  hits@10 {:.4}'.format(lp_results['mrr'],
                                                                                                    lp_results['h@1'],
                                                                                                    lp_results['h@3'],
                                                                                                    lp_results['h@10']))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_params': model.model_params,
                'loss_val': mean_loss,
                'loss_log': loss_dict},
                result_dir + '/rgvae_dict.pt')
                
            if mean_loss > old_loss:
                print('Validation loss diverging:{:.3} vs. {:.3}'.format(mean_loss, best_loss))
            if mean_loss < best_loss:
                best_loss = mean_loss
            old_loss = mean_loss


if __name__ == "__main__":
    pass
