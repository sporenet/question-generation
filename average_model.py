import argparse
import pickle
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', required=True,
                    help='input word2vec model directory')
parser.add_argument('--pickle', '-p', required=True,
                    help='pickle file (optional)')
parser.add_argument('--output', '-o', default='result',
                    help='Directory to output the sent2vec, doc2vec model file')
args = parser.parse_args()

logging.info("Reading word vector model file...")

with open(args.input + '/word2vec_avg.model', 'r') as f:
    ss = f.readline().split()
    n_word, n_units = int(ss[0]), int(ss[1])
    word2index = {}
    index2word = {}
    wword = np.empty((n_word, n_units), dtype=np.float32)
    for i, line in enumerate(f):
        ss = line.split(',')
        assert len(ss) == n_units + 1
        word = ss[0]
        word2index[word] = i
        index2word[i] = word
        wword[i] = np.array([float(s) for s in ss[1:]], dtype=np.float32)

sword = np.sqrt((wword * wword).sum(1))
wword /= sword.reshape((sword.shape[0], 1))

logging.info("Reading pickle file...")

wiki_data = pickle.load(open(args.pickle, 'rb'))
docsentdic = wiki_data.doc_sent_pkl
sentwordrawdic = wiki_data.sent_word_raw_pkl

logging.info("Calculating sentence vector...")

with open(args.output + '/sent2vec_avg.model', 'w') as f:
    n_sent = len(sentwordrawdic)
    f.write('%d %d\n' % (n_sent, n_units))
    sent2index = {}
    index2sent = {}
    wsent = np.empty((n_sent, n_units), dtype=np.float32)
    for i, s in enumerate(sentwordrawdic):
        vs = np.zeros(n_units, dtype=np.float32)
        line = sentwordrawdic[s]
        line = line.rstrip()
        words = line.split(' ')
        words = [w.lower() for w in words]
        for w in words:
            vw = wword[word2index[w]]
            vs += vw
        vs = vs / len(words)
        vs = vs.astype(dtype=np.float32)
        sent2index[s] = i
        index2sent[i] = s
        wsent[i] = vs
        f.write(s + ',')
        f.write(','.join([str(x) for x in vs]) + '\n')

ssent = np.sqrt((wsent * wsent).sum(1))
wsent /= ssent.reshape((ssent.shape[0], 1))

logging.info("Calculating document vector...")

with open(args.output + '/doc2vec_avg.model', 'w') as f:
    n_doc = len(docsentdic)
    f.write('%d %d\n' % (n_doc, n_units))
    for d in docsentdic:
        vd = np.zeros(n_units, dtype=np.float32)
        for s in docsentdic[d]:
            vs = wsent[sent2index[s]]
            vd += vs
        vd = vd / len(docsentdic[d])
        vd = vd.astype(dtype=np.float32)
        f.write(d + ',')
        f.write(','.join([str(x) for x in vd]) + '\n')



