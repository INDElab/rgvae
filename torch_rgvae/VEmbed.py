import torch
import torch.nn as nn
from lp_utils import d
from torch_rgvae.decoders import DistMult



class Venco(nn.Module):
    def __init__(self, n_e: int, n_r: int, z_dim: int, var: bool=True):
        super().__init__()
        self.z_dim = z_dim
        self.n_e = n_e
        self.n_r = n_r
        self.var = var
        # self.model_params = args

        # Encoder
        self.e_embed = nn.Embedding(n_e, 2*self.z_dim)
        self.r_embed = nn.Embedding(n_r, 2*self.z_dim)

    def encode(self, s, r, o):
        """
        """
        bs = s.shape[0]
        # reparametrization
        z_s = self.reparameterize(self.e_embed(s))
        z_r = self.reparameterize(self.r_embed(r))
        z_o = self.reparameterize(self.e_embed(o))
        return (z_s, z_r, z_o)

    def reparameterize(self, mean_logvar):
        """
        Reparametrization trick.
        """
        self.mean = mean = mean_logvar[:, :, :self.z_dim]
        self.logvar = logvar = mean_logvar[:, :, self.z_dim:]
        if self.var:
            eps = torch.normal(torch.zeros_like(mean), std=1.).to(d())
        else:
            eps = 1.
        return eps * torch.exp(logvar * .5) + mean


class VLinkPredictor(nn.Module):
    """
    Link prediction model with no message passing
    Outputs raw (linear) scores for the given triples.
    """

    def __init__(self, triples, n, r, embedding=512, decoder='distmult', edropout=None, rdropout=None, init=0.85,
                 biases=False, init_method='uniform', init_parms=(-1.0, 1.0), reciprocal=False):

        super().__init__()

        assert triples.dtype == torch.long

        self.n, self.r = n, r
        self.e = embedding
        self.reciprocal = reciprocal

        # self.entities  = nn.Parameter(torch.FloatTensor(n, self.e))
        # initialize(self.entities, init_method, init_parms)
        # self.relations = nn.Parameter(torch.FloatTensor(r, self.e))
        # initialize(self.relations, init_method, init_parms)

        # if reciprocal:
        #     self.relations_backward = nn.Parameter(torch.FloatTensor(r, self.e).uniform_(-init, init))
        #     initialize(self.relations, init_method, init_parms)

        self.encoder = Venco(n, r, embedding)

        if decoder == 'distmult':
            self.decoder = DistMult(embedding)

        else:
            raise Exception()

        self.edo = None
        self.rdo = None

    def forward(self, s, p, o, recip=None):
        """
        Takes a batch of triples in s, p, o indices, and computes their scores.
        If s, p and o have more than one dimension, and the same shape, the resulting score
        tensor has that same shape.
        If s, p and o have more than one dimension and mismatching shape, they are broadcast together
        and the score tensor has the broadcast shape. If broadcasting fails, the method fails. In order to trigger the
        correct optimizations, it's best to ensure that all tensors have the same dimensions.
        :param s:
        :param p:
        :param o:
        :param recip: prediction mode, if this is a reciprocal model. 'head' for head prediction, 'tail' for tail
            prediction, 'eval' for the average of both (i.e. for final scoring).
        :return:
        """

        assert recip in [None, 'head', 'tail', 'eval']
        assert self.reciprocal or (recip is None), 'Predictor must be set to model reciprocal relations for recip to be set'
        if self.reciprocal and recip is None:
            recip = 'eval'

        scores = 0

        if recip is None or recip == 'tail':
            modes = [True] # forward only
        elif recip == 'head':
            modes = [False] # backward only
        elif recip == 'eval':
            modes = [True, False]
        else:
            raise Exception(f'{recip=}')

        for forward in modes:

            si, pi, oi = (s, p, o) if forward else (o, p, s)

            z_s, z_p, z_o = self.encoder.encode(s, p, o)

            # nodes = self.entities
            # relations = self.relations if forward else self.relations_backward

            # # Apply dropout
            # nodes = nodes if self.edo is None else self.edo(nodes)
            # relations = relations if self.rdo is None else self.rdo(relations)

            scores = scores + self.decoder(z_s, z_p, z_o)
            # -- We let the decoder handle the broadcasting

            # if self.biases:
            #     pb = self.pbias if forward else self.pbias_bw

            #     scores = scores + (self.sbias[si] + pb[pi] + self.obias[oi] + self.gbias)

        if self.reciprocal:
            scores = scores / len(modes)

        return scores

    def penalty(self, rweight, p, which):

        # TODO implement weighted penalty

        if which == 'entities':
            params = [self.encoder.e_embed]
        elif which == 'relations':
            params = [self.encoder.r_embed]
        else:
            raise Exception()

        if p % 2 == 1:
            params = [p.abs() for p in params]

        return (rweight / p) * sum([(p ** p).sum() for p in params])
