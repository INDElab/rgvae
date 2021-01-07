
import torch
import numpy as np
from torch_rgvae.GVAE import GVAE
from torch_rgvae.GCVAE import GCVAE
from lp_utils import *
import pickle as pkl
import wandb


def eval_triple(triples, exist_triples, key_entities):
    """
    """
    n_true = 0.
    new_triples = []
    for triple in triples:
        if triple[0] in key_entities or triple[0] in key_entities:
            n_true += 1
            if triple not in exist_triples:
                new_triples.append(triple)
    p_new = len(new_triples)/n_true if n_true > 0 else 0.
    wandb.log({'n_true': n_true, 'n_new': len(new_triples), 'n_filtered': len(triples)})
    return (n_true/len(triples), p_new, new_triples)
            

def eval_generation(model, i2n, i2r, all_triples, n_eval: int=1000, key_type: str='people'):
    """
    Experiment: Generate triples from random latent space signals
                Filter based on if the predicate including the key type
                Check if subject entity is of key type 
    """
    with open('data/fb15k/e2t_dict.pkl', 'rb') as f:
        entity_text_dict = pkl.load(f)   

    with open('data/fb15k/e2type_dict.pkl', 'rb') as f:
        entity_type_dict = pkl.load(f)

    i2keep = [index for (index,rel) in enumerate(i2r) if key_type in rel]
    filt_triples = set()

    for i, triple in enumerate(all_triples):
        if triple[1] in i2keep:
                filt_triples.add(triple)
    n2keep = []
    n_blacklist = []
    for n, entity in enumerate(i2n):
        try:
            if key_type in entity_type_dict[entity]:
                n2keep.append(n)
        except:
            n_blacklist.append(n)
            print('No types for: {}'.format(entity_text_dict[entity][0]))

    print('From {} entities, {} contain the keyword {}, or {:.3f}%.'.format(len(i2n), len(n2keep), key_type, 100*len(n2keep)/len(i2n)))
    print('From {} predicates, {} contain the keyword {}, or {:.3f}%.'.format(len(i2r), len(i2keep), key_type, 100*len(i2keep)/len(i2r)))
    wandb.log({'total_n': len(i2n), 'key_n': len(n2keep), 'total_r': len(i2r), 'key_n': len(i2keep), 'key_type': key_type})

    triples = list()
    breaker = 0
    while breaker < n_eval:
        signal = torch.randn((1, model.z_dim), device=d())
        pred_dense = matrix2triple(model.sample(signal))
        for i_triple in pred_dense:
            if i_triple[1] in i2keep:
                triples.append(i_triple)
                breaker += 1
    
    triples = translate_triple(triples, i2n, i2r, entity_text_dict)
    p_true, p_new, new_triples =  eval_triple(triples, all_triples, n2keep)
    return (p_true, p_new, translate_triple(new_triples, i2n, i2r, entity_text_dict)), triples
            


if __name__ == "__main__":
    
    pass
