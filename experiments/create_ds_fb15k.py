"""
Obsolete since it exceeds working memory.
"""
import torch
import numpy as np
from utils.lp_utils import *
from tqdm import tqdm


def create_ds_fb15k(n: int, batch_size: int):
    """
    Converts the fb15k dataset into sparse matrix representation.
    Args:
        n: Number of triples to be used in one graph.
        batch_size: Number of graphs to concatenates per batch.
    """
    data_file = 'data/fb15k_n{}_bs{}.pkl'.format(n, batch_size)

    # Check for data folder and eventually create.
    if not os.path.isdir('data'):
        os.mkdir('data')

    if os.path.isfile(data_file):
        with open(data_file, "rb") as fp:
            print('Loading dataset..')
            train_set, test_set, dims = pickle.load(fp)
    else:
        print('Creating dataset..')
        (n2i, i2n), (r2i, i2r), train, test, all_triples = load_link_prediction_data('fb15k')
        d_n = len(n2i)
        d_e = len(r2i)
        train_set = list()
        test_set = list()
        i = 0
        pbar = tqdm(total=len(train)+len(test))
        for (data_in, data_set) in [(train, train_set), (test, test_set)]:
            while len(data_in) > i * batch_size:
                batch_a = list()
                batch_e = list()
                batch_f = list()
                for ii in range(batch_size):
                    pbar.update(1)
                    a = i*batch_size + ii*n
                    if a+n > len(data_in):
                        break
                    (A, E, F) = triple2matrix(data_in[a:a+n], d_n, d_e)
                    if A.shape[1] == 1:
                        print('Weirdo here:', data_in[a:a+n],a)
                    batch_a.append(A)
                    batch_e.append(E)
                    batch_f.append(F)
                data_set.append([torch.cat(batch_a, dim=0),torch.cat(batch_e, dim=0),torch.cat(batch_f, dim=0)])
                i += 1
        pbar.close()
        dims = (d_n,d_e)
        with open(data_file, "wb") as fp:
                pickle.dump([train_set, test_set, dims], fp)
    return (train_set, test_set, dims)
