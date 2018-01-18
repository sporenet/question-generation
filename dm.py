import argparse
import collections

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
from chainer import training
from chainer.training import extensions

import logging
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

class DistributedMemory(chainer.Chain):

    # def __init__(self, n_vocab, n_docs, n_units, loss_func):
    #     super(DistributedMemory, self).__init__(
    #         embed=F.EmbedID(
    #             n_vocab+n_docs, n_units, initialW=I.Uniform(1. / n_units)),
    #         loss_func=loss_func,
    #     )

    def __init__(self, n_vocab, n_sents, n_docs, n_units, loss_func):
        super(DistributedMemory, self).__init__(
            embed=F.EmbedID(
                n_vocab+n_sents+n_docs, n_units, initialW=I.Uniform(1. / n_units)),
            loss_func=loss_func
        )

    # def __call__(self, x, doc, context):
    #     d = self.embed(doc)
    #     e = self.embed(context)
    #     h = F.sum(e, axis=1) + d
    #     h = h * (1. / (context.data.shape[1] + 1))
    #     loss = self.loss_func(h, x)
    #     reporter.report({'loss': loss}, self)
    #     return loss

    def __call__(self, center_word, center_sent, sent, doc, context_word, context_sent):
        window_word = context_word.data.shape
        window_sent = context_sent.data.shape
        shape_doc = doc.data.shape
        shape_sent = sent.data.shape

        s = F.reshape(sent, (shape_sent[0], 1))
        s = F.concat((s, context_word), axis=1)
        s = F.reshape(s, (shape_sent[0] * (window_word[1] + 1),))
        t = F.broadcast_to(center_word[:, None], (shape_sent[0], window_word[1] + 1))
        t = F.reshape(t, (shape_sent[0] * (window_word[1] + 1),))
        s = self.embed(s)
        loss = self.loss_func(s, t)

        x = F.reshape(doc, (shape_doc[0], 1))
        x = F.concat((x, context_sent), axis=1)
        x = F.reshape(x, (shape_doc[0] * (window_sent[1] + 1),))
        y = F.broadcast_to(center_sent[:, None], (shape_doc[0], window_sent[1] + 1))
        y = F.reshape(y, (shape_doc[0] * (window_sent[1] + 1),))
        x = self.embed(x)
        loss += self.loss_func(x, y)

        # s = self.embed(sent)
        # t = self.embed(context_word)
        # g = F.sum(t, axis=1) + s
        # g = g * (1. / (context_word.data.shape[1] + 1))
        # loss = self.loss_func(g, center_word)
        #
        # d = self.embed(doc)
        # e = self.embed(context_sent)
        # h = F.sum(e, axis=1) + d
        # h = h * (1. / (context_sent.data.shape[1] + 1))
        # loss += self.loss_func(h, center_sent)
        #
        reporter.report({'loss': loss}, self)
        return loss
