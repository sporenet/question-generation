import argparse
import collections
import pickle
import codecs
import math

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import chainer.optimizers as O
from chainer import training
from chainer.training import extensions

from dm import DistributedMemory
from dbow import DistributedBoW
from iterator import WindowIterator

import logging
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

# For executing in pycharm
import os
os.environ["PATH"] += ":/home/junghyuk/.local/cuda-8.0/bin:/home/junghyuk/.local/cuda-8.0/bin"

from dataset import WikiData
from dataset import GloVeData

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', required=True,
                    help='input text file')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--unit', '-u', default=200, type=int,
                    help='number of units')
parser.add_argument('--window-word', '-ww', default=5, type=int,
                    help='window size of word')
parser.add_argument('--window-sent', '-ws', default=3, type=int,
                    help='window size of sentence')
parser.add_argument('--batchsize', '-b', type=int, default=1000,
                    help='learning minibatch size')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--negative-size', default=5, type=int,
                    help='number of negative samples')
parser.add_argument('--out-type', '-ot', choices=['hsm', 'ns', 'original'],
                    default='hsm',
                    help='output model type ("hsm": hierarchical softmax, '
                         '"ns": negative sampling, "original": no approximation)')
parser.add_argument('--output', '-o', default='result',
                    help='Directory to output the result')
parser.add_argument('--glove', '-glv', default=None,
                    help='Glove word vector file for word vector initialization')
parser.add_argument('--pickle', '-p', default=None,
                    help='pickle file (optional)')
args = parser.parse_args()

WORDVEC_PATH = args.output + '/word2vec.model'
SENTVEC_PATH = args.output + '/sent2vec.model'
DOCVEC_PATH = args.output + '/doc2vec.model'
GLOVE_PATH = args.glove
if args.pickle:
    WIKIDATA_PATH = args.pickle
else:
    WIKIDATA_PATH = args.output + '/wiki_data.pkl'

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    cuda.check_cuda_available()

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('Window(word): {}'.format(args.window_word))
print('Window(sentence): {}'.format(args.window_sent))
print('Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('Training model: {}'.format('dm'))
print('Output type: {}'.format(args.out_type))
print('')

xp = cuda.cupy if args.gpu >= 0 else np


class SoftmaxCrossEntropyLoss(chainer.Chain):

    def __init__(self, n_in, n_out):
        super(SoftmaxCrossEntropyLoss, self).__init__(
            out=L.Linear(n_in, n_out, initialW=0),
        )

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.out(x), t)


def convert(batch, device):
    center_word, center_sent, sent, doc, context_word, context_sent = batch
    if device >= 0:
        center_word = cuda.to_gpu(center_word)
        center_sent = cuda.to_gpu(center_sent)
        sent = cuda.to_gpu(sent)
        doc = cuda.to_gpu(doc)
        context_word = cuda.to_gpu(context_word)
        context_sent = cuda.to_gpu(context_sent)
    return center_word, center_sent, sent, doc, context_word, context_sent

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()

if args.pickle:
    wiki_data = pickle.load(open(WIKIDATA_PATH, 'rb'))
else:
    wiki_data = WikiData(file_name=args.input, ww=args.window_word, ws=args.window_sent)
    wiki_data.load_data()
    pickle.dump(wiki_data, open(WIKIDATA_PATH, 'wb'))

len_doc_label = 0
for j in wiki_data.l_doc_sent:
    len_doc_label += j
doc_label = xp.zeros((len_doc_label,))
cursor_doc_label = 0
for i, j in enumerate(wiki_data.l_doc_sent):
    tmp = xp.broadcast_to(xp.array(wiki_data.doc_title[i]), (j,))
    doc_label[cursor_doc_label:cursor_doc_label+j] = tmp
    cursor_doc_label += j

len_sent_label = 0
for j in wiki_data.l_sent_word:
    len_sent_label += j
sent_label = xp.zeros((len_sent_label,))
cursor_sent_label = 0
for i, j in enumerate(wiki_data.l_sent_word):
    tmp = xp.broadcast_to(xp.array(wiki_data.sent_title[i]), (j,))
    sent_label[cursor_sent_label:cursor_sent_label+j] = tmp
    cursor_sent_label += j

index2doc = {did: doc for doc, did in six.iteritems(wiki_data.docs)}
index2sent = {sid: sent for sent, sid in six.iteritems(wiki_data.sents)}
index2word = {wid: word for word, wid in six.iteritems(wiki_data.words)}

counts = collections.Counter(wiki_data.doc_title)
counts.update(collections.Counter(wiki_data.doc_sent))
counts.update(collections.Counter(wiki_data.sent_word))
n_words = len(wiki_data.words)
n_sents = len(wiki_data.sents)
n_docs = len(wiki_data.docs)

doc_sent = xp.asarray(wiki_data.doc_sent).astype(xp.int32)
sent_word = xp.asarray(wiki_data.sent_word).astype(xp.int32)
doc_label = doc_label.astype(xp.int32)
sent_label = sent_label.astype(xp.int32)

logging.info("Finished reading data")
logging.info("Number of vocabulary: %d" % n_words)
logging.info("Number of sentence: %d" % n_sents)
logging.info("Number of document: %d" % n_docs)

if args.out_type == 'hsm':
    HSM = L.BinaryHierarchicalSoftmax
    tree = HSM.create_huffman_tree(counts)
    loss_func = HSM(args.unit, tree)
    loss_func.W.data[...] = 0
elif args.out_type == 'ns':
    cs = [counts[w] for w in range(len(counts))]
    loss_func = L.NegativeSampling(args.unit, cs, args.negative_size)
    loss_func.W.data[...] = 0
elif args.out_type == 'original':
    loss_func = SoftmaxCrossEntropyLoss(args.unit, n_words + n_sents + n_docs)
else:
    raise Exception('Unknown output type: {}'.format(args.out_type))

model = DistributedMemory(n_words, n_sents, n_docs, args.unit, loss_func)

if args.gpu >= 0:
    model.to_gpu()

if args.glove:
    logging.info("Loading glove vectors...")
    glove_data = GloVeData(file_name=GLOVE_PATH)

    glove_data.load_data()
    for i in index2word.keys():
        vec = glove_data.get_vector(index2word[i])
        if vec is not None:
            model.embed.W.data[i] = xp.asarray(glove_data.get_vector(index2word[i]))

logging.info("Finished building model")

optimizer = O.Adam()
optimizer.setup(model)

train_iter = WindowIterator(doc_sent, sent_word, doc_label, sent_label, args.window_word, args.window_sent, args.batchsize)
updater = training.StandardUpdater(
    train_iter, optimizer, converter=convert, device=args.gpu
)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.output)
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
trainer.extend(extensions.ProgressBar(update_interval=100))
trainer.run()

w = cuda.to_cpu(model.embed.W.data)

with codecs.open(WORDVEC_PATH, 'w', 'utf-8') as f:
    f.write('%d %d\n' % (len(index2word), args.unit))
    word_indices = list(index2word.keys())
    word_embedding = [w[i] for i in word_indices]
    for i, wi in zip(word_indices, word_embedding):
        v = ','.join(map(str, wi))
        f.write('%s,%s\n' % (index2word[i], v))

with codecs.open(SENTVEC_PATH, 'w', 'utf-8') as f:
    f.write('%d %d\n' % (len(index2sent), args.unit))
    sent_indices = list(index2sent.keys())
    sent_embedding = [w[i] for i in sent_indices]
    for i, wi in zip(sent_indices, sent_embedding):
        v = ','.join(map(str, wi))
        f.write('%s,%s\n' % (index2sent[i], v))

with codecs.open(DOCVEC_PATH, 'w', 'utf-8') as f:
    f.write('%d %d\n' % (len(index2doc), args.unit))
    doc_indices = list(index2doc.keys())
    doc_embedding = [w[i] for i in doc_indices]
    for i, wi in zip(doc_indices, doc_embedding):
        v = ','.join(map(str, wi))
        f.write('%s,%s\n' % (index2doc[i], v))
