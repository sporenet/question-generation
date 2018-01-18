#!/usr/bin/env python
import numpy as np
import six
import argparse
import codecs
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', required=True,
                    help='directory of input model file')
parser.add_argument('--model', '-m', required=True,
                    help='models you want to compare with (dd ds dw ss sw ww)')
parser.add_argument('--count', '-c', type=int, default=10,
                    help='number of search result to show')
args = parser.parse_args()

model = ['', '']

for i in range(len(args.model)):
    m = args.model[i]
    if m == 'd':
        model[i] = args.input + '/doc2vec.model'
    elif m == 's':
        model[i] = args.input + '/sent2vec.model'
    elif m == 'w':
        model[i] = args.input + '/word2vec.model'
    else:
        print("Please input d, s or w")
        exit()

if args.model == 'ds':
    dictionary = pickle.load(open(args.input + '/doc_sent.pkl', 'rb'))
    additional_dictionary = pickle.load(open(args.input + '/sent_word_raw.pkl', 'rb'))
elif args.model == 'dw':
    dictionary = pickle.load(open(args.input + '/doc_word.pkl', 'rb'))
    additional_dictionary = {}
elif args.model == 'sw':
    dictionary = pickle.load(open(args.input + '/sent_word.pkl', 'rb'))
    additional_dictionary = pickle.load(open(args.input + '/sent_word_raw.pkl', 'rb'))
elif args.model == 'ss':
    dictionary = {}
    additional_dictionary = pickle.load(open(args.input + '/sent_word_raw.pkl', 'rb'))
else:
    dictionary = {}
    additional_dictionary = {}

with codecs.open(model[0], 'r', 'utf-8') as f:
    ss = f.readline().split()
    n_vocab, n_units = int(ss[0]), int(ss[1])
    vocab12index = {}
    index2vocab1 = {}
    w1 = np.empty((n_vocab, n_units), dtype=np.float32)
    for i, line in enumerate(f):
        ss = line.split(',')
        assert len(ss) == n_units + 1
        vocab1 = ss[0]
        vocab12index[vocab1] = i
        index2vocab1[i] = vocab1
        w1[i] = np.array([float(s) for s in ss[1:]], dtype=np.float32)

with codecs.open(model[1], 'r', 'utf-8') as f:
    ss = f.readline().split()
    n_vocab, n_units = int(ss[0]), int(ss[1])
    vocab22index = {}
    index2vocab2 = {}
    w2 = np.empty((n_vocab, n_units), dtype=np.float32)
    for i, line in enumerate(f):
        ss = line.split(',')
        assert len(ss) == n_units + 1
        vocab2 = ss[0]
        vocab22index[vocab2] = i
        index2vocab2[i] = vocab2
        w2[i] = np.array([float(s) for s in ss[1:]], dtype=np.float32)

s1 = np.sqrt((w1 * w1).sum(1))
w1 /= s1.reshape((s1.shape[0], 1))  # normalize

s2 = np.sqrt((w2 * w2).sum(1))
w2 /= s2.reshape((s2.shape[0], 1))  # normalize

try:
    while True:
        q = six.moves.input('>> ')
        if q == 'quit;':
            break
        if q not in vocab12index:
            print('"{0}" is not found'.format(q))
            continue
        if q in additional_dictionary.keys():
            print(additional_dictionary[q])
        v = w1[vocab12index[q]]
        similarity = w2.dot(v)
        print('query: {}'.format(q))
        count = 0
        for i in (-similarity).argsort():
            if np.isnan(similarity[i]):
                continue
            if index2vocab2[i] == q:
                continue
            if index2vocab2[i] in ['#PAD_SENT#', '#PAD_WORD#']:
                continue
            if len(dictionary) != 0 and index2vocab2[i] not in dictionary[q]:
                continue
            if index2vocab2[i] in additional_dictionary.keys():
                if len(additional_dictionary[index2vocab2[i]].split()) < 10:
                    continue
                else:
                    print(additional_dictionary[index2vocab2[i]])
            print('{0}: {1}'.format(index2vocab2[i], similarity[i]))
            count += 1
            if count == args.count:
                break

except EOFError:
    pass
