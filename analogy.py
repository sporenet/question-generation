#!/usr/bin/env python
import numpy as np
import six
import argparse
import codecs

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', required=True,
                    help='model file')
parser.add_argument('--count', '-c', type=int, default=10,
                    help='number of search result to show')
args = parser.parse_args()

with codecs.open(args.model, 'r', 'utf-8') as f:
    ss = f.readline().split()
    n_vocab, n_units = int(ss[0]), int(ss[1])
    vocab2index = {}
    index2vocab = {}
    w = np.empty((n_vocab, n_units), dtype=np.float32)
    for i, line in enumerate(f):
        ss = line.split(',')
        assert len(ss) == n_units + 1
        vocab = ss[0]
        vocab2index[vocab] = i
        index2vocab[i] = vocab
        w[i] = np.array([float(s) for s in ss[1:]], dtype=np.float32)

s = np.sqrt((w * w).sum(1))
w /= s.reshape((s.shape[0], 1))  # normalize

try:
    print("Usage: seoul - korea + japan (must split tokens by whitespace)")
    while True:
        q = six.moves.input('>> ')
        if q == 'quit;':
            break
        vars = q.split()

        if len(vars) != 5:
            print("Please input valid expression")
            continue

        v0 = vars[0]
        v1 = vars[2]
        v2 = vars[4]

        if v0 not in vocab2index:
            print('"{0}" is not found'.format(v0))
            continue
        if v1 not in vocab2index:
            print('"{0}" is not found'.format(v1))
            continue
        if v2 not in vocab2index:
            print('"{0}" is not found'.format(v2))
            continue\

        op0 = vars[1]
        op1 = vars[3]

        if op0 == '+':
            v = [x + y for x, y in zip(w[vocab2index[v0]], w[vocab2index[v1]])]
        elif op0 == '-':
            v = [x - y for x, y in zip(w[vocab2index[v0]], w[vocab2index[v1]])]
        else:
            print('Invalid operator "{0}"'.format(op0))
            continue

        if op1 == '+':
            v = [x + y for x, y in zip(v, w[vocab2index[v2]])]
        elif op1 == '-':
            v = [x - y for x, y in zip(v, w[vocab2index[v2]])]
        else:
            print('Invalid operator "{0}"'.format(op1))
            continue

        similarity = w.dot(v)
        print('query: {}'.format(q))
        count = 0
        for i in (-similarity).argsort():
            if np.isnan(similarity[i]):
                continue
            if index2vocab[i] == q:
                continue
            if index2vocab[i] in ['#PAD_SENT#', '#PAD_WORD#']:
                continue
            print('{0}: {1}'.format(index2vocab[i], similarity[i]))
            count += 1
            if count == args.count:
                break

except EOFError:
    pass
