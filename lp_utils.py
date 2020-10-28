"""
Utile functions for link prediction.
"""
import gzip, os, pickle, tqdm
import torch
import numpy as np


def locate_file(filepath):
    directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return directory + '/rgvae/' + filepath


def load_strings(file):
    """ Read triples from file """
    with open(file, 'r') as f:
        return [line.split() for line in f]

def load_link_prediction_data(name, use_test_set=False, limit=None):
    """
    Load knowledge graphs for relation Prediction  experiment.
    Source: https://github.com/pbloem/gated-rgcn/blob/1bde7f28af8028f468349b2d760c17d5c908b58b/kgmodels/data.py#L218
    :param name: Dataset name ('aifb', 'am', 'bgs' or 'mutag')
    :param use_test_set: If true, load the canonical test set, otherwise load validation set from file.
    :param limit: If set, only the first n triples are used.
    :return: Relation prediction test and train sets:
              - train: list of edges [subject, predicate object]
              - test: list of edges [subject, predicate object]
              - all_triples: sets of tuples (subject, predicate object)
    """

    if name.lower() == 'fb15k':
        train_file = locate_file('data/fb15k/train.txt')
        val_file = locate_file('data/fb15k/valid.txt')
        test_file = locate_file('data/fb15k/test.txt')
    elif name.lower() == 'fb15k-237':
        train_file = locate_file('data/fB15k-237/train.txt')
        val_file = locate_file('data/fB15k-237/valid.txt')
        test_file = locate_file('data/fB15k-237/test.txt')
    elif name.lower() == 'wn18':
        train_file = locate_file('data/wn18/train.txt')
        val_file = locate_file('data/wn18/valid.txt')
        test_file = locate_file('data/wn18/test.txt')
    elif name.lower() == 'wn18rr':
        train_file = locate_file('data/wn18rr/train.txt')
        val_file = locate_file('data/wn18rr/valid.txt')
        test_file = locate_file('data/wn18rr/test.txt')
    else:
        raise ValueError(f'Could not find \'{name}\' dataset')

    train = load_strings(train_file)
    val = load_strings(val_file)
    test = load_strings(test_file)

    if use_test_set:
        train = train + val
    else:
        test = val

    if limit:
        train = train[:limit]
        test = test[:limit]

    # Mappings for nodes (n) and relations (r)
    nodes, rels = set(), set()
    for i in train + val + test:
      if len(i) < 3:
        print(i)
    for s, p, o in train + test:
        nodes.add(s)
        rels.add(p)
        nodes.add(o)

    i2n, i2r = list(nodes), list(rels)
    n2i, r2i = {n: i for i, n in enumerate(nodes)}, {r: i for i, r in enumerate(rels)}

    all_triples = set()
    for s, p, o in train + test:
        all_triples.add((n2i[s], r2i[p], n2i[o]))

    train = [[n2i[st[0]], r2i[st[1]], n2i[st[2]]] for st in train]
    test = [[n2i[st[0]], r2i[st[1]], n2i[st[2]]] for st in test]

    return (n2i, i2n), (r2i, i2r), train, test, all_triples

def triple2matrix(triples, max_n: int, max_r: int):
    """
    Transforms triples into matrix form.
    Params:
        triples: set of sparse triples
        max_n: total count of nodes
        max_t: total count of relations
    Outputs the A,E,F matrices for the input triples,
    """
    # An exception for single triples.
    n_list = list(dict.fromkeys([triple[0] for triple in triples]))+list(dict.fromkeys([triple[2] for triple in triples]))
    n_dict =  dict(zip(n_list, np.arange(len(n_list))))
    n = 2*len(triples)     # All matrices must be of same size

    # The empty first dimension is to stacking into batches.
    
    A = torch.zeros((1,n,n))
    E = torch.zeros((1,n,n,max_r))
    F = torch.zeros((1,n,max_n))

    for (s, r, o) in triples:
        i_s, i_o = n_dict[s], n_dict[o]
        A[0,i_s,i_o] = 1
        E[0,i_s,i_o,r] = 1
        F[0,i_s,s] = 1
        F[0,i_o,o] = 1
    return (A, E, F)

def matrix2triple(graph):
    """
    Converts a sparse graph back to triple from.
    Args:
        graph: Graph consisting of A, E, F matrix
    returns a set of triples/one triple.
    """
    A, E, F = graph
    a = A.squeeze().detach().cpu().numpy()
    e = E.squeeze().detach().cpu().numpy()
    f = F.squeeze().detach().cpu().numpy()
    
    s, o = np.where(a == 1)
    _, n_index = np.where(f == 1)
    _, _, r_index = np.where(e == 1)
    
    triples = list()
    for i in range(len(s)):
        cho = np.where(e[s[i],o[i],:] == 1)[0]
        if cho.size > 0:
            r = np.random.choice(cho[0])
            triple = (n_index[s[i]], r, n_index[o[i]])
            triples.append(triple)
    return triples


def translate_triple(triples, i2n, i2r):
    """
    Translate an indexed triple back to text.
    Args:
    ....
    """
    triples_text = list()
    for triple in triples:
        (s,r,o) = triple
        triples_text.append((i2n[s], i2r[r], i2n[0]))
    return triples_text


def batch_t2m(i, data_set, batch_size, n, d_n, d_e):
    """
    Converts batches of triples into matrix form.
    Args:
        i: the current batch number.
        data_set: ice scream.
        n: number of triples per. matrix
        d_n: total node count.
        d_e: total edge attribute count.
    returns the batched matrices A, E, F.
    """
    batch_a = list()
    batch_e = list()
    batch_f = list()
    for ii in range(batch_size):
        a = i*batch_size + ii*n
        if a+n > len(data_set):
            break
        (A, E, F) = triple2matrix(data_set[a:a+n], d_n, d_e)
        if A.shape[1] == 1:
            print('here', data_in[a:a+n],a)
        batch_a.append(A)
        batch_e.append(E)
        batch_f.append(F)

    return [torch.cat(batch_a, dim=0),torch.cat(batch_e, dim=0),torch.cat(batch_f, dim=0)]



if __name__ == "__main__":
    (n2i, i2n), (r2i, i2r), train, test, all_triples = load_link_prediction_data('fb15k')
    max_n = len(n2i)
    max_r = len(r2i)
    (A, E, F) = triple2matrix(train[111:211], max_n, max_r)
    print(A)