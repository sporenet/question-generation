import argparse
import pickle
import math
import numpy as np
from utils import word_tokenize
from statistics import mean
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import logging
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

lemmatizer = WordNetLemmatizer()

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', required=True,
                    help='directory of input model file')
parser.add_argument('--pickle', '-p', required=True,
                    help='input quiz pickle file')
parser.add_argument('--output', '-o', required=True,
                    help='file name of output quizzes')
parser.add_argument('--threshold', '-t', type=float, default=0.5,
                    help='threshold of quiz score')
args = parser.parse_args()


WIKIDATA_PATH = args.input + '/wiki_data.pkl'
INDEXTERM_PATH = args.input + '/index_term.pkl'

W1 = [1 / 3, 1 / 3, 1 / 3]     # sentence / gap / distractor
W2 = [1 / 3, 1 / 3, 1 / 3]     # index_term / document_title / TF-IDF

logging.info("Reading pickle file...")

# Read dictionary
wiki_data = pickle.load(open(WIKIDATA_PATH, 'rb'))
docsentdic = wiki_data.doc_sent_pkl
docworddic = wiki_data.doc_word_pkl
sentworddic = wiki_data.sent_word_pkl
sentwordrawdic = wiki_data.sent_word_raw_pkl
postagdic = wiki_data.pos_tag_pkl

logging.info("Reading index term...")
index_term = pickle.load(open(INDEXTERM_PATH, 'rb'))
#index_term_lower = [t.decode('utf-8').lower() for t in index_term]
index_term_lower = [t.lower() for t in index_term]
index_term_lower_lemmatized = [lemmatizer.lemmatize(t) for t in index_term_lower]
index_term_lower_lemmatized = [t for t in index_term_lower_lemmatized if t not in stopwords.words('english')]

logging.info("Constructing document words...")
docwords = {}

for doc in docsentdic:
    sents_in_doc = docsentdic[doc]
    doc_words_lower_lemmatized = []
    for sent in sents_in_doc:
        doc_words_lower_lemmatized += [lemmatizer.lemmatize(t) for t in word_tokenize(sentwordrawdic[sent], lower=True)]
    doc_words_lower_lemmatized = [t for t in doc_words_lower_lemmatized if t not in stopwords.words('english')]
    docwords[doc] = doc_words_lower_lemmatized


def score_dt_s(s, d):
    doc_title_words = d.replace('_', ' ').replace('/', ' ').split()
    doc_title_words_lower = [t.lower() for t in doc_title_words]
    doc_title_words_lower_lemmatized = [lemmatizer.lemmatize(t) for t in doc_title_words_lower]
    doc_title_words_lower_lemmatized = [t for t in doc_title_words_lower_lemmatized if
                                        t not in stopwords.words('english')]
    sent_words = word_tokenize(s, lower=True)
    sent_words_lemmatized = [lemmatizer.lemmatize(t) for t in sent_words]

    return len([x for x in doc_title_words_lower_lemmatized if x in sent_words_lemmatized]) / len(doc_title_words_lower_lemmatized)


def score_dt_w(w, d):
    doc_title_words = d.replace('_', ' ').replace('/', ' ').split()
    doc_title_words_lower = [t.lower() for t in doc_title_words]
    doc_title_words_lower_lemmatized = [lemmatizer.lemmatize(t) for t in doc_title_words_lower]
    doc_title_words_lower_lemmatized = [t for t in doc_title_words_lower_lemmatized if
                                        t not in stopwords.words('english')]

    return int(lemmatizer.lemmatize(w) in doc_title_words_lower_lemmatized)


def score_dt(quiz):
    doc = quiz.document
    sent = quiz.sentence.replace('______', '')
    gap = quiz.gap
    dist = quiz.distractors

    score = W1[0] * score_dt_s(sent, doc) + W1[1] * score_dt_w(gap, doc) + W1[2] * mean([score_dt_w(x, doc) for x in dist])
    return score


def score_it_s(s, d):
    sents_in_doc = docsentdic[d]
    doc_words_lower_lemmatized = []
    for sent in sents_in_doc:
        doc_words_lower_lemmatized += [lemmatizer.lemmatize(t) for t in word_tokenize(sentwordrawdic[sent], lower=True)]
    doc_words_lower_lemmatized = [t for t in doc_words_lower_lemmatized if t not in stopwords.words('english')]
    sent_words_lower_lemmatized = [lemmatizer.lemmatize(t) for t in word_tokenize(s, lower=True)]
    sent_words_lower_lemmatized = [t for t in sent_words_lower_lemmatized if t not in stopwords.words('english')]

    return len([t for t in sent_words_lower_lemmatized if t in index_term_lower_lemmatized]) / len([t for t in doc_words_lower_lemmatized if t in index_term_lower_lemmatized])


def score_it_w(w):
    return int(lemmatizer.lemmatize(w) in index_term_lower_lemmatized)


def score_it(quiz):
    doc = quiz.document
    sent = quiz.sentence.replace('______', '')
    gap = quiz.gap
    dist = quiz.distractors

    score = W1[0] * score_it_s(sent, doc) + W1[1] * score_it_w(gap) + W1[2] * mean([score_it_w(x) for x in dist])
    return score


def tf(w, d):
    sents_in_doc = docsentdic[d]
    doc_words_lower_lemmatized = []
    for sent in sents_in_doc:
        doc_words_lower_lemmatized += [lemmatizer.lemmatize(t) for t in word_tokenize(sentwordrawdic[sent], lower=True)]
    doc_words_lower_lemmatized = [t for t in doc_words_lower_lemmatized if t not in stopwords.words('english')]

    return math.log(1 + (doc_words_lower_lemmatized.count(w) / len(doc_words_lower_lemmatized)))


def idf(w):
    n_containing = 0
    for doc in docwords:
        n_containing += int(w in docwords[doc])

    return 1 / n_containing if n_containing != 0 else 0


def score_tfidf_w(w, d):
    return tf(w, d) * idf(w)


def score_tfidf_s(s, d):
    sent_words_lower_lemmatized = [lemmatizer.lemmatize(t) for t in word_tokenize(s, lower=True)]
    sent_words_lower_lemmatized = [t for t in sent_words_lower_lemmatized if t not in stopwords.words('english')]

    return mean([score_tfidf_w(w, d) for w in sent_words_lower_lemmatized])


def score_tfidf(quiz):
    doc = quiz.document
    sent = quiz.sentence.replace('______', '')
    gap = lemmatizer.lemmatize(quiz.gap)
    dist = [lemmatizer.lemmatize(t) for t in quiz.distractors]

    score = W1[0] * score_tfidf_s(sent, doc) + W1[1] * score_tfidf_w(gap, doc) + W1[2] * mean([score_tfidf_w(x, doc) for x in dist])
    return score


logging.info("Calculating quiz score...")
quizzes = pickle.load(open(args.pickle, 'rb'))
score_tfidfs = []
score_dts = []
score_its = []

for quiz in quizzes:
    score_tfidfs.append(score_tfidf(quiz))
    score_dts.append(score_dt(quiz))
    score_its.append(score_it(quiz))

# Min-max normalization
score_tfidfs = [(x - min(score_tfidfs)) / (max(score_tfidfs) - min(score_tfidfs)) for x in score_tfidfs]
score_dts_norm = [(x - min(score_dts)) / (max(score_dts) - min(score_dts)) for x in score_dts]
score_its_norm = [(x - min(score_its)) / (max(score_its) - min(score_its)) for x in score_its]

score_quizzes = [W2[0] * it + W2[1] * dt + W2[2] * tfidf for it, dt, tfidf in zip(score_its, score_dts, score_tfidfs)]
score_quizzes = np.array(score_quizzes)

with open(args.output, 'w') as f:
    for i in score_quizzes.argsort()[::-1]:
        if score_quizzes[i] < args.threshold:
            break
        q = quizzes[i]
        f.write(q.document + '\n')
        print(q.document)
        f.write(q.sentence + '\n')
        print(q.sentence)
        f.write(q.gap + ' ')
        f.write(' '.join(q.distractors) + '\n')
        print(q.gap + ' ' + ' '.join(q.distractors))
        print(score_quizzes[i])

