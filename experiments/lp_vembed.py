from torch_rgvae.VEmbed import VLinkPredictor
from utils.lp_utils import d, tic, toc, get_slug, load_link_prediction_data, truedicts
from utils.embed_util import util
from ranger import Ranger

import torch, wandb

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import random, sys, tqdm, math, random, os
from datetime import date

from tqdm import trange

from argparse import ArgumentParser

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# import multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter
from torch_rgvae.losses import kl_divergence


"""
Experiment to see if bias terms help link prediction
"""

def corrupt(batch, n):
    """
    Corrupts the negatives of a batch of triples (in place).
    Randomly corrupts either heads or tails
    :param batch_size:
    :param n: nr of nodes in the graph
    :return:
    """
    bs, ns, _ = batch.size()

    # new entities to insert
    corruptions = torch.randint(size=(bs * ns,),low=0, high=n, dtype=torch.long, device=d(batch))

    # boolean mask for entries to corrupt
    mask = torch.bernoulli(torch.empty(size=(bs, ns, 1), dtype=torch.float, device=d(batch)).fill_(0.5)).to(torch.bool)
    zeros = torch.zeros(size=(bs, ns, 1), dtype=torch.bool, device=d(batch))
    mask = torch.cat([mask, zeros, ~mask], dim=2)

    batch[mask] = corruptions

def corrupt_one(batch, candidates, target):
    """
    Corrupts the negatives of a batch of triples (in place).
    Corrupts either only head or only tails
    :param batch_size:
    :param n: nr of nodes in the graph
    :param target: 0 for head, 1 for predicate, 2 for tail
    :return:
    """
    bs, ns, _ = batch.size()

    # new entities to insert
    #corruptions = torch.randint(size=(bs * ns,),low=0, high=n, dtype=torch.long, device=d(batch))
    corruptions = torch.tensor(random.choices(candidates, k=bs*ns),  dtype=torch.long, device=d(batch)).view(bs, ns)

    batch[:, :, target] = corruptions

def prt(to_p, end='\n'):
    print(to_p + end)

def train_lp_vembed(n_e, n_r, train, test, alltriples, beta: int, epochs: int, batch_size: int, result_dir: str, test_batch: int=5, eval_int: int=3):
    """
    Source: pbloem/embed
    """
    # Fix some hyperparameters
    repeats = 1
    sched = True
    check_simple = True
    negative_rate = [10,0,10]         # No neg sampling in VAE
    limit_negatives = True
    loss_fn = 'bce'
    reciprocal = True
    reg_exp = 2
    reg_eweight = None
    reg_rweight = None
    eval_size = None
    corrupt_global = True
    bias = True
    patience = 1
    result_file = result_dir + '/lp_log.txt'

    lr = 1e-4
    k = 11

    model = VLinkPredictor(torch.tensor(list(alltriples)), n_e, n_r, embedding=512, decoder='distmult', edropout=None, rdropout=None, init=0.85, biases=False, init_method='uniform', init_parms=(-1.0, 1.0), reciprocal=reciprocal)
    wandb.watch(model)

    tbw = SummaryWriter(log_dir=result_dir)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_mrrs = []
    truedict = truedicts(alltriples)
    train = torch.tensor(train).to(d())
    test = torch.tensor(test).to(d())

    subjects   = torch.tensor(list({s for s, _, _ in train}), dtype=torch.long, device=d())
    predicates = torch.tensor(list({p for _, p, _ in train}), dtype=torch.long, device=d())
    objects    = torch.tensor(list({o for _, _, o in train}), dtype=torch.long, device=d())
    ccandidates = (subjects, predicates, objects)

    with open(result_file, 'w') as f:
        print(n_e, 'nodes', file=f)
        print(n_r, 'relations', file=f)
        print(train.size(0), 'training triples', file=f)
        print(test.size(0), 'test triples', file=f)
        print(train.size(0) + test.size(0), 'total triples', file=f)

    for r in tqdm.trange(repeats) if repeats > 1 else range(repeats):

        if torch.cuda.is_available():
            prt('Using CUDA.')
            model.cuda()

        # if arg.opt == 'adam':
        #     opt = torch.optim.Adam(model.parameters(), lr=arg.lr)
        # elif arg.opt == 'adamw':
        #     opt = torch.optim.AdamW(model.parameters(), lr=arg.lr)
        # elif arg.opt == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.15953749294870845)
        # elif arg.opt == 'sgd':
        #     opt = torch.optim.SGD(model.parameters(), lr=arg.lr, nesterov=True, momentum=arg.momentum)
        # else:
        #     raise Exception()

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=patience, optimizer=optimizer, mode='max', factor=0.95, threshold=0.0001) \
            if sched else None
        #-- defaults taken from libkge

        # nr of negatives sampled
        weight = None

        seen = 0
        for e in range(epochs):

            seeni, sumloss = 0, 0.0
            tforward = tbackward = 0
            rforward = rbackward = 0
            tprep = tloss = 0
            tic()

            for fr in trange(0, train.size(0), batch_size):
                to = min(train.size(0), fr + batch_size)

                model.train(True)

                optimizer.zero_grad()

                positives = train[fr:to].to(d())

                for ctarget in [0, 1, 2]: # which part of the triple to corrupt
                    ng = negative_rate[ctarget]

                    if ng > 0:

                        with torch.no_grad():
                            bs, _ = positives.size()

                            tic()
                            if limit_negatives:
                                cand = ccandidates[ctarget]
                                mx = cand.size(0)
                                idx = torch.empty(bs, ng, dtype=torch.long, device=d()).random_(0, mx)
                                corruptions = cand[idx]
                            else:
                                mx = n_r if ctarget == 1 else n_e
                                corruptions = torch.empty(bs, ng, dtype=torch.long, device=d()).random_(0, mx)
                            tprep += toc()

                            s, p, o = positives[:, 0:1], positives[:, 1:2], positives[:, 2:3]
                            if ctarget == 0:
                                s = torch.cat([s, corruptions], dim=1)
                            if ctarget == 1:
                                p = torch.cat([p, corruptions], dim=1)
                            if ctarget == 2:
                                o = torch.cat([o, corruptions], dim=1)

                            # -- NB: two of the index vectors s, p o are now size (bs, 1) and the other is (bs, ng+1)
                            #    We will let the model broadcast these to give us a score tensor of (bs, ng+1)
                            #    In most cases we can optimize the decoder to broadcast late for better speed.

                            if loss_fn == 'bce':
                                labels = torch.cat([torch.ones(bs, 1, device=d()), torch.zeros(bs, ng, device=d())], dim=1)
                            elif loss_fn == 'ce':
                                labels = torch.zeros(bs, dtype=torch.long, device=d())
                                # -- CE loss treats the problem as a multiclass classification problem: for a positive triple,
                                #    together with its k corruptions, identify which is the true triple. This is always triple 0.
                                #    (It may seem like the model could easily cheat by always choosing triple 0, but the score
                                #    function is order equivariant, so it can't choose by ordering.)

                        recip = None if not reciprocal else ('head' if ctarget == 0 else 'tail')
                        # -- We use the tail relations if the target is the relation (usually p-corruption is not used)

                        tic()
                        out = model.forward(s, p, o)
                        tforward += toc()

                        assert out.size() == (bs, ng + 1), f'{out.size()=} {(bs, ng + 1)=}'

                        tic()
                        if loss_fn == 'bce':
                            out = F.sigmoid(out)
                            recon_loss = F.binary_cross_entropy_with_logits(out, labels, weight=weight)
                            wandb.log({"recon_loss": recon_loss})
                            reg_loss = kl_divergence(model.encoder.mean, model.encoder.logvar)
                            loss = torch.mean(beta * reg_loss - recon_loss)
                        # elif loss_fn == 'ce':
                        #     loss = F.cross_entropy(out, labels)
                        wandb.log({"loss": loss.item()})
                        assert not torch.isnan(loss), 'Loss has become NaN'

                        sumloss += float(loss.item())
                        seen += bs; seeni += bs
                        tloss += toc()

                        tic()
                        loss.backward()
                        tbackward += toc()
                        # No step yet, we accumulate the gradients over all corruptions.
                        # -- this causes problems with modules like batchnorm, so be careful when porting.

                tic()
                regloss = None
                # if reg_eweight is not None:
                #     regloss = model.penalty(which='entities', p=reg_exp, rweight=reg_eweight)

                # if reg_rweight is not None:
                #     regloss = model.penalty(which='relations', p=reg_exp, rweight=reg_rweight)
                # rforward += toc()

                # tic()
                # if regloss is not None:
                #     sumloss += float(regloss.item())
                #     regloss.backward()
                #     wandb.log({"regloss": regloss})
                # rbackward += toc()

                optimizer.step()

            if e == 0:
                print('\n pred: forward {tforward:.4}, backward {tbackward:.4}')
                print (f'           prep {tprep:.4}, loss {tloss:.4}')
                print (f' total: {toc():.4}')
                # -- NB: these numbers will not be accurate for GPU runs unless CUDA_LAUNCH_BLOCKING is set to 1

            # Evaluate
            if ((e+1) % eval_int == 0) or e == epochs - 1:

                with torch.no_grad():

                    model.train(False)

                    if eval_size is None:
                        testsub = test
                    else:
                        testsub = test[random.sample(range(test.size(0)), k=eval_size)]

                    mrr, hits, ranks = util.eval(
                        model=model, valset=testsub, truedicts=truedict, n=n_e, batch_size=test_batch, verbose=True)

                    # if check_simple: # double-check using a separate, slower implementation
                    #     mrrs, hitss, rankss = util.eval_simple(
                    #         model=model, valset=testsub, alltriples=alltriples, n=n_e, verbose=True)
                        # assert ranks == rankss
                        # assert mrr == mrrs
                    with open(result_file, 'a+') as f:
                        print(f'epoch {e}: MRR {mrr:.4}\t hits@1 {hits[0]:.4}\t  hits@3 {hits[1]:.4}\t  hits@10 {hits[2]:.4}', file=f)

                    print(f'epoch {e}: MRR {mrr:.4}\t hits@1 {hits[0]:.4}\t  hits@3 {hits[1]:.4}\t  hits@10 {hits[2]:.4}')
                    tbw.add_scalar('biases/mrr', mrr, e)
                    tbw.add_scalar('biases/h@1', hits[0], e)
                    tbw.add_scalar('biases/h@3', hits[1], e)
                    tbw.add_scalar('biases/h@10', hits[2], e)
                    wandb.log({"mrr": mrr, "h@1": hits[0], "h@3": hits[1], "h@10": hits[2]})

                    if sched is not None:
                        sched.step(mrr) # reduce lr if mrr stalls

        test_mrrs.append(mrr)
    with open(result_file, 'a+') as f:
        print('training finished.', file=f)
    print('training finished.')

    temrrs = torch.tensor(test_mrrs)
    with open(result_file, 'a+') as f:
        print(f'mean test MRR    {temrrs.mean():.3} ({temrrs.std():.3})  \t{test_mrrs}', file=f)
    print(f'mean test MRR    {temrrs.mean():.3} ({temrrs.std():.3})  \t{test_mrrs}')

