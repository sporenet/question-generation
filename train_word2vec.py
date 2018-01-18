import argparse
import collections
import numpy as np
import six
import pickle

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L
import chainer.optimizers as O
from chainer import reporter
from chainer import training
from chainer.training import extensions

from dataset import WikiDataW2V
from dataset import GloVeData

import logging
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

# For executing in pycharm
import os
os.environ["PATH"] += ":/home/junghyuk/.local/cuda-8.0/bin:/home/junghyuk/.local/cuda-8.0/bin"

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', required=True,
                    help='input text file')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--unit', '-u', default=100, type=int,
                    help='number of units')
parser.add_argument('--window', '-w', default=5, type=int,
                    help='window size')
parser.add_argument('--batchsize', '-b', type=int, default=1000,
                    help='learning minibatch size')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--model', '-m', choices=['skipgram', 'cbow'],
                    default='skipgram',
                    help='model type ("skipgram", "cbow")')
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


WORDVEC_PATH = args.output + '/word2vec_avg.model'
GLOVE_PATH = args.glove
if args.pickle:
    WIKIDATA_PATH = args.pickle
else:
    WIKIDATA_PATH = args.output + '/wiki_data_avg.pkl'

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    cuda.check_cuda_available()

xp = cuda.cupy if args.gpu >= 0 else np

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('Window: {}'.format(args.window))
print('Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('Training model: {}'.format(args.model))
print('Output type: {}'.format(args.out_type))
print('')


class ContinuousBoW(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func):
        super(ContinuousBoW, self).__init__(
            embed=F.EmbedID(
                n_vocab, n_units, initialW=I.Uniform(1. / n_units)),
            loss_func=loss_func,
        )

    def __call__(self, x, context):
        e = self.embed(context)
        h = F.sum(e, axis=1) * (1. / context.data.shape[1])
        loss = self.loss_func(h, x)
        reporter.report({'loss': loss}, self)
        return loss


class SkipGram(chainer.Chain):

    def __init__(self, n_vocab, n_units, loss_func):
        super(SkipGram, self).__init__(
            embed=L.EmbedID(
                n_vocab, n_units, initialW=I.Uniform(1. / n_units)),
            loss_func=loss_func,
        )

    def __call__(self, x, context):
        e = self.embed(context)
        shape = e.data.shape
        x = F.broadcast_to(x[:, None], (shape[0], shape[1]))
        e = F.reshape(e, (shape[0] * shape[1], shape[2]))
        x = F.reshape(x, (shape[0] * shape[1],))
        loss = self.loss_func(e, x)
        reporter.report({'loss': loss}, self)
        return loss


class SoftmaxCrossEntropyLoss(chainer.Chain):

    def __init__(self, n_in, n_out):
        super(SoftmaxCrossEntropyLoss, self).__init__(
            out=L.Linear(n_in, n_out, initialW=0),
        )

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.out(x), t)


class WindowIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, window, batch_size, repeat=True):
        self.dataset = np.array(dataset, np.int32)
        self.window = window
        self.batch_size = batch_size
        self._repeat = repeat

        self.order = np.random.permutation(
            len(dataset) - window * 2).astype(np.int32)
        self.order += window
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size
        position = self.order[i: i_end]
        w = np.random.randint(self.window - 1) + 1
        offset = np.concatenate([np.arange(-w, 0), np.arange(1, w + 1)])
        pos = position[:, None] + offset[None, :]
        context = self.dataset.take(pos)
        center = self.dataset.take(position)

        if i_end >= len(self.order):
            np.random.shuffle(self.order)
            self.epoch += 1
            self.is_new_epoch = True
            self.current_position = 0
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return center, context

    @property
    def epoch_detail(self):
        return self.epoch + float(self.current_position) / len(self.order)

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            serializer('_order', self._order)


def convert(batch, device):
    center, context = batch
    if device >= 0:
        center = cuda.to_gpu(center)
        context = cuda.to_gpu(context)
    return center, context


if args.gpu >= 0:
    cuda.get_device(args.gpu).use()

if args.pickle:
    wiki_data = pickle.load(open(WIKIDATA_PATH, 'rb'))
else:
    wiki_data = WikiDataW2V(file_name=args.input)
    wiki_data.load_data()
    pickle.dump(wiki_data, open(WIKIDATA_PATH, 'wb'))

index2word = {wid: word for word, wid in six.iteritems(wiki_data.words)}

counts = collections.Counter(wiki_data.all_words)
n_vocab = len(wiki_data.words)

logging.info("Finished reading data")
logging.info("Number of vocabulary: %d" % n_vocab)

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
    loss_func = SoftmaxCrossEntropyLoss(args.unit, n_vocab)
else:
    raise Exception('Unknown output type: {}'.format(args.out_type))

if args.model == 'skipgram':
    model = SkipGram(n_vocab, args.unit, loss_func)
elif args.model == 'cbow':
    model = ContinuousBoW(n_vocab, args.unit, loss_func)
else:
    raise Exception('Unknown model type: {}'.format(args.model))

if args.gpu >= 0:
    model.to_gpu()

glove_data = GloVeData(file_name=GLOVE_PATH)
glove_data.load_data()

for i in index2word.keys():
    vec = glove_data.get_vector(index2word[i])
    if vec is not None:
        model.embed.W.data[i] = xp.asarray(glove_data.get_vector(index2word[i]))

optimizer = O.Adam()
optimizer.setup(model)

train_iter = WindowIterator(wiki_data.all_words, args.window, args.batchsize)
updater = training.StandardUpdater(
    train_iter, optimizer, converter=convert, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.output)
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss']))
trainer.extend(extensions.ProgressBar())
trainer.run()

with open(WORDVEC_PATH, 'w') as f:
    f.write('%d %d\n' % (len(index2word), args.unit))
    w = cuda.to_cpu(model.embed.W.data)
    for i, wi in enumerate(w):
        v = ','.join(map(str, wi))
        f.write('%s,%s\n' % (index2word[i], v))

