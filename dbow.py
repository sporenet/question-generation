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


class DistributedBoW(chainer.Chain):

    # def __init__(self, n_vocab, n_docs, n_units, loss_func):
    #     super(DistributedBoW, self).__init__(
    #         embed=F.EmbedID(
    #             n_vocab+n_docs, n_units, initialW=I.Uniform(1. / n_units)),
    #         loss_func=loss_func,
    #     )

    def __init__(self, n_vocab, n_sents, n_docs, n_units, loss_func):
        super(DistributedBoW, self).__init__(
            embed=F.EmbedID(
                n_vocab+n_sents+n_docs, n_units, initialW=I.Uniform(1. / n_units)),
            loss_func=loss_func
        )

    # # TODO: loss_function 계산을 gpu가 아니라 cpu로 할 방법 없나? (memory 부족...)
    # def to_gpu(self, device=None):
    #     with cuda.get_device(device):
    #         self.loss_func.W.to_gpu()

    # def __call__(self, x, doc, context, train_word=True):
    #     window = context.data.shape
    #     shape = doc.data.shape
    #     d = F.broadcast_to(doc[:, None], (shape[0], window[1]))
    #     d = F.reshape(d, (shape[0] * window[1],))
    #     e = F.reshape(context, (shape[0] * window[1],))
    #     d = self.embed(d)
    #     loss = self.loss_func(d, e)
    #
    #     if train_word:
    #         x = F.broadcast_to(x[:, None], (shape[0], window[1]))
    #         x = F.reshape(x, (shape[0] * window[1],))
    #         x = self.embed(x)
    #         loss += self.loss_func(x, e)
    #
    #     reporter.report({'loss': loss}, self)
    #     return loss

    def __call__(self, center_word, center_sent, sent, doc, context_word, context_sent):
        window_word = context_word.data.shape
        window_sent = context_sent.data.shape
        shape_doc = doc.data.shape
        shape_sent = sent.data.shape

        s = F.broadcast_to(sent[:, None], (shape_sent[0], window_word[1]))
        s = F.reshape(s, (shape_sent[0] * window_word[1],))
        t = F.reshape(context_word, (shape_sent[0] * window_word[1],))
        s = self.embed(s)
        loss = self.loss_func(s, t)

        y = F.broadcast_to(center_word[:, None], (shape_sent[0], window_word[1]))
        y = F.reshape(y, (shape_sent[0] * window_word[1],))
        y = self.embed(y)
        loss += self.loss_func(y, t)

        d = F.broadcast_to(doc[:, None], (shape_doc[0], window_sent[1]))
        d = F.reshape(d, (shape_doc[0] * window_sent[1],))
        e = F.reshape(context_sent, (shape_doc[0] * window_sent[1],))
        d = self.embed(d)
        loss += self.loss_func(d, e)

        x = F.broadcast_to(center_sent[:, None], (shape_doc[0], window_sent[1]))
        x = F.reshape(x, (shape_doc[0] * window_sent[1],))
        x = self.embed(x)
        loss += self.loss_func(x, e)

        reporter.report({'loss': loss}, self)
        return loss
