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

from dm import DistributedMemory
from dbow import DistributedBoW

import logging
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)


class WindowIterator(chainer.dataset.Iterator):

    # def __init__(self, text, label, window, batch_size, repeat=True):
    #     self.text = np.array(text, np.int32)
    #     self.label = np.array(label, np.int32)
    #     self.window = window
    #     self.batch_size = batch_size
    #     self._repeat = repeat
    #
    #     self.order = np.random.permutation(
    #         len(text) - window * 2).astype(np.int32)
    #     self.order += window
    #     self.current_position = 0
    #     self.epoch = 0
    #     self.is_new_epoch = False

    def __init__(self, doc_sent, sent_word, doc_label, sent_label, window_word, window_sent, batch_size, repeat=True):
        self.doc_sent = doc_sent
        self.sent_word = sent_word
        self.doc_label = doc_label
        self.sent_label = sent_label

        self.window_word = window_word
        self.window_sent = window_sent
        self.batch_size_word = round(len(sent_word) / (len(doc_sent) + len(sent_word)) * batch_size)
        self.batch_size_sent = batch_size - self.batch_size_word

        self.order_word = np.random.permutation(len(sent_word) - window_word * 2).astype(np.int32)
        self.order_word += window_word

        self.order_sent = np.random.permutation(len(doc_sent) - window_sent * 2).astype(np.int32)
        self.order_sent += window_sent

        self.curr_pos_word = 0
        self.curr_pos_sent = 0

        self.epoch = 0
        self.is_new_epoch = False

        self._repeat = repeat

    # def __next__(self):
    #     if not self._repeat and self.epoch > 0:
    #         raise StopIteration
    #
    #     i = self.current_position
    #     i_end = i + self.batch_size
    #     position = self.order[i: i_end]
    #     w = np.random.randint(self.window - 1) + 1
    #     offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
    #     pos = position[:, None] + offset[None, :]
    #     context = self.text.take(pos)
    #     doc = self.label.take(position)
    #     center = self.text.take(position)
    #
    #     if i_end >= len(self.order):
    #         np.random.shuffle(self.order)
    #         self.epoch += 1
    #         self.is_new_epoch = True
    #         self.current_position = 0
    #     else:
    #         self.is_new_epoch = False
    #         self.current_position = i_end
    #
    #     return center, doc, context

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        ww = np.random.randint(self.window_word - 1) + 1
        ws = np.random.randint(self.window_sent - 1) + 1

        # For sent_word
        isw = self.curr_pos_word
        isw_end = isw + self.batch_size_word
        position = self.order_word[isw: isw_end]
        offset = np.concatenate([np.arange(-ww, 0), np.arange(1, ww + 1)])
        pos = position[:, None] + offset[None, :]
        context_word = self.sent_word.take(pos)
        sent = self.sent_label.take(position)
        center_word = self.sent_word.take(position)

        # For doc_sent
        ids = self.curr_pos_sent
        ids_end = ids + self.batch_size_sent
        position = self.order_sent[ids: ids_end]
        offset = np.concatenate([np.arange(-ws, 0), np.arange(1, ws + 1)])
        pos = position[:, None] + offset[None, :]
        context_sent = self.doc_sent.take(pos)
        doc = self.doc_label.take(position)
        center_sent = self.doc_sent.take(position)

        # For word and sent, it will end at the same iteration
        if ids_end >= len(self.order_sent):
            np.random.shuffle(self.order_word)
            np.random.shuffle(self.order_sent)
            self.epoch += 1
            self.is_new_epoch = True
            self.curr_pos_word = 0
            self.curr_pos_sent = 0
        else:
            self.is_new_epoch = False
            self.curr_pos_word = isw_end
            self.curr_pos_sent = ids_end

        return center_word, center_sent, sent, doc, context_word, context_sent

    # @property
    # def epoch_detail(self):
    #     return self.epoch + float(self.current_position) / len(self.order)
    #
    # def serialize(self, serializer):
    #     self.current_position = serializer('current_position',
    #                                        self.current_position)
    #     self.epoch = serializer('epoch', self.epoch)
    #     self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
    #     if self._order is not None:
    #         serializer('_order', self._order)

    @property
    def epoch_detail(self):
        return self.epoch + float(self.curr_pos_word) / len(self.order_word)

    def serialize(self, serializer):
        self.curr_pos_word = serializer('curr_pos_word',
                                        self.curr_pos_word)
        self.curr_pos_sent = serializer('curr_pos_sent',
                                        self.curr_pos_sent)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch',
                                       self.is_new_epoch)
        if self._order_word is not None:
            serializer('_order_word', self._order_word)
        if self._order_sent is not None:
            serializer('_order_sent', self._order_sent)