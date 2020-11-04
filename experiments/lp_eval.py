import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import random, sys, tqdm, math, random, os
from tqdm import trange
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
"""
Experiment to see if bias terms help link prediction
"""

EPSILON = 0.000000001

global repeats

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

def prt(str, end='\n'):
    if repeats == 1:
        print(str, end=end)

def go(arg):

    global repeats
    repeats = arg.repeats

    tbdir = arg.tb_dir if arg.tb_dir is not None else os.path.join('./runs', get_slug(arg))[:250]
    tbw = SummaryWriter(log_dir=tbdir)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_mrrs = []

    train, val, test, (n2i, i2n), (r2i, i2r) = \
        embed.load(arg.name)

    # set of all triples (for filtering)
    alltriples = set()
    for s, p, o in torch.cat([train, val, test], dim=0):
        s, p, o = s.item(), p.item(), o.item()

        alltriples.add((s, p, o))

    truedicts = util.truedicts(alltriples)

    if arg.final:
        train, test = torch.cat([train, val], dim=0), test
    else:
        train, test = train, val

    subjects   = torch.tensor(list({s for s, _, _ in train}), dtype=torch.long, device=d())
    predicates = torch.tensor(list({p for _, p, _ in train}), dtype=torch.long, device=d())
    objects    = torch.tensor(list({o for _, _, o in train}), dtype=torch.long, device=d())
    ccandidates = (subjects, predicates, objects)

    print(len(i2n), 'nodes')
    print(len(i2r), 'relations')
    print(train.size(0), 'training triples')
    print(test.size(0), 'test triples')
    print(train.size(0) + test.size(0), 'total triples')

    for r in tqdm.trange(repeats) if repeats > 1 else range(repeats):

        """
        Define model
        """
        model = embed.LinkPredictor(
            triples=train, n=len(i2n), r=len(i2r), embedding=arg.emb, biases=arg.biases,
            edropout = arg.edo, rdropout=arg.rdo, decoder=arg.decoder, reciprocal=arg.reciprocal,
            init_method=arg.init_method, init_parms=arg.init_parms)

        if torch.cuda.is_available():
            prt('Using CUDA.')
            model.cuda()

        if arg.opt == 'adam':
            opt = torch.optim.Adam(model.parameters(), lr=arg.lr)
        elif arg.opt == 'adamw':
            opt = torch.optim.AdamW(model.parameters(), lr=arg.lr)
        elif arg.opt == 'adagrad':
            opt = torch.optim.Adagrad(model.parameters(), lr=arg.lr)
        elif arg.opt == 'sgd':
            opt = torch.optim.SGD(model.parameters(), lr=arg.lr, nesterov=True, momentum=arg.momentum)
        else:
            raise Exception()

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=arg.patience, optimizer=opt, mode='max', factor=0.95, threshold=0.0001) \
            if arg.sched else None
        #-- defaults taken from libkge

        # nr of negatives sampled
        weight = torch.tensor([arg.nweight, 1.0], device=d()) if arg.nweight else None

        seen = 0
        for e in range(arg.epochs):

            seeni, sumloss = 0, 0.0
            tforward = tbackward = 0
            rforward = rbackward = 0
            tprep = tloss = 0
            tic()

            for fr in trange(0, train.size(0), arg.batch):
                to = min(train.size(0), fr + arg.batch)

                model.train(True)

                opt.zero_grad()

                positives = train[fr:to].to(d())

                for ctarget in [0, 1, 2]: # which part of the triple to corrupt
                    ng = arg.negative_rate[ctarget]

                    if ng > 0:

                        with torch.no_grad():
                            bs, _ = positives.size()

                            tic()
                            if arg.limit_negatives:
                                cand = ccandidates[ctarget]
                                mx = cand.size(0)
                                idx = torch.empty(bs, ng, dtype=torch.long, device=d()).random_(0, mx)
                                corruptions = cand[idx]
                            else:
                                mx = len(i2r) if ctarget == 1 else len(i2n)
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

                            if arg.loss == 'bce':
                                labels = torch.cat([torch.ones(bs, 1, device=d()), torch.zeros(bs, ng, device=d())], dim=1)
                            elif arg.loss == 'ce':
                                labels = torch.zeros(bs, dtype=torch.long, device=d())
                                # -- CE loss treats the problem as a multiclass classification problem: for a positive triple,
                                #    together with its k corruptions, identify which is the true triple. This is always triple 0.
                                #    (It may seem like the model could easily cheat by always choosing triple 0, but the score
                                #    function is order equivariant, so it can't choose by ordering.)

                        recip = None if not arg.reciprocal else ('head' if ctarget == 0 else 'tail')
                        # -- We use the tail relations if the target is the relation (usually p-corruption is not used)

                        tic()
                        out = model(s, p, o, recip=recip)
                        tforward += toc()

                        assert out.size() == (bs, ng + 1), f'{out.size()=} {(bs, ng + 1)=}'

                        tic()
                        if arg.loss == 'bce':
                            loss = F.binary_cross_entropy_with_logits(out, labels, weight=weight, reduction=arg.lred)
                        elif arg.loss == 'ce':
                            loss = F.cross_entropy(out, labels, reduction=arg.lred)

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
                if arg.reg_eweight is not None:
                    regloss = model.penalty(which='entities', p=arg.reg_exp, rweight=arg.reg_eweight)

                if arg.reg_rweight is not None:
                    regloss = model.penalty(which='relations', p=arg.reg_exp, rweight=arg.reg_rweight)
                rforward += toc()

                tic()
                if regloss is not None:
                    sumloss += float(regloss.item())
                    regloss.backward()
                rbackward += toc()

                opt.step()

                tbw.add_scalar('biases/train_loss', float(loss.item()), seen)

            if e == 0:
                print(f'\n pred: forward {tforward:.4}, backward {tbackward:.4}')
                print (f'   reg: forward {rforward:.4}, backward {rbackward:.4}')
                print (f'           prep {tprep:.4}, loss {tloss:.4}')
                print (f' total: {toc():.4}')
                # -- NB: these numbers will not be accurate for GPU runs unless CUDA_LAUNCH_BLOCKING is set to 1

            # Evaluate
            if ((e+1) % arg.eval_int == 0) or e == arg.epochs - 1:

                with torch.no_grad():

                    model.train(False)

                    testsub = test

                    mrr, hits, ranks = util.eval(
                        model=model, valset=testsub, truedicts=truedicts, n=len(i2n),
                        batch_size=arg.test_batch, verbose=True)

                    if arg.check_simple: # double-check using a separate, slower implementation
                        mrrs, hitss, rankss = util.eval_simple(
                            model=model, valset=testsub, alltriples=alltriples, n=len(i2n), verbose=True)

                        assert ranks == rankss
                        assert mrr == mrrs

                    print(f'epoch {e}: MRR {mrr:.4}\t hits@1 {hits[0]:.4}\t  hits@3 {hits[1]:.4}\t  hits@10 {hits[2]:.4}')

                    tbw.add_scalar('biases/mrr', mrr, e)
                    tbw.add_scalar('biases/h@1', hits[0], e)
                    tbw.add_scalar('biases/h@3', hits[1], e)
                    tbw.add_scalar('biases/h@10', hits[2], e)

                    if sched is not None:
                        sched.step(mrr) # reduce lr if mrr stalls

        test_mrrs.append(mrr)

    print('training finished.')

    temrrs = torch.tensor(test_mrrs)
    print(f'mean test MRR    {temrrs.mean():.3} ({temrrs.std():.3})  \t{test_mrrs}')