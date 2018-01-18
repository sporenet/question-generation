import codecs
import numpy as np
import argparse
import six
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model1', '-m1', required=True,
                    help='model file 1')
parser.add_argument('--model2', '-m2', required=True,
                    help='model file 2')
parser.add_argument('--count', '-c', type=int, default=10,
                    help='number of result to show')
parser.add_argument('--out', '-o', default='figure.png',
                    help='output file name')
args = parser.parse_args()

with codecs.open(args.model1, 'r', 'utf-8') as f:
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

with codecs.open(args.model2, 'r', 'utf-8') as f:
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

print("Usage: korea japan china or korea (must split tokens by whitespace)")
while True:
    q = six.moves.input('>> ')
    if q == 'quit;':
        break

    query_words = q.split()
    X = []

    if len(query_words) == 1:
        if q not in vocab12index:
            print('"{0}" is not found'.format(q))
            continue
        X.append(w1[vocab12index[q]])

        v = w1[vocab12index[query_words[0]]]
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
            word = index2vocab2[i]
            query_words.append(word)
            X.append(w2[vocab22index[word]])
            count += 1
            if count == args.count:
                break
    else:
        for word in query_words:
            X.append(w2[vocab22index[word]])

    # do PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    X = pca.transform(X)
    xs = X[:, 0]
    ys = X[:, 1]

    # draw
    plt.clf()
    plt.scatter(xs, ys, marker='o')
    for i, w in enumerate(query_words):
        plt.annotate(
            w,
            xy=(xs[i], ys[i]), xytext=(3, 3),
            textcoords='offset points', ha='left', va='top',
        )

    plt.savefig(args.out)
    print('ok.')
