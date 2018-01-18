import os
import argparse
import numpy as np
import pickle
import random
import nltk
import wordnet as wn
from statistics import mean
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from dataset import Quiz
from utils import word_tokenize

import logging
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

lemmatizer = WordNetLemmatizer()

SENTENCE_MIN_LENGTH = 10
CHOICES_SHUFFLE = False
ENABLE_BLANK = True

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', required=True,
                    help='directory of input model file')
parser.add_argument('--num-sent', '-ns', type=int, default=4,
                    help='number of sentences per document')
parser.add_argument('--num-gap', '-ng', type=int, default=2,
                    help='number of gaps per sentence')
parser.add_argument('--num-dist', '-nd', type=int, default=3,
                    help='number of distractors per quiz')
parser.add_argument('--output', '-o', required=True,
                    help='file name of output quizzes')
parser.add_argument('--num-quiz', '-nq', type=int, default=100000000,
                    help='number of quizzes you want to generate')
parser.add_argument('--model', '-m', choices=['joint', 'avg', 'none'], default='joint',
                    help='type of model (\'joint\' or \'avg\' or \'none\'')
args = parser.parse_args()

hand_stopwords = ['figure', 'figures', 'table', 'tables', 'chapter', 'section', 'chapters', 'sections']

model = args.model
if model in ['joint', 'none']:
    model = ''
else:
    model = '_' + model
DOCVEC_PATH = args.input + '/doc2vec' + model + '.model'
SENTVEC_PATH = args.input + '/sent2vec' + model + '.model'
WORDVEC_PATH = args.input + '/word2vec' + model + '.model'
WIKIDATA_PATH = args.input + '/wiki_data.pkl'
INDEXTERM_PATH = args.input + '/index_term.pkl'

logging.info("Reading pickle file...")

# Read dictionary
wiki_data = pickle.load(open(WIKIDATA_PATH, 'rb'))
docsentdic = wiki_data.doc_sent_pkl
docworddic = wiki_data.doc_word_pkl
sentworddic = wiki_data.sent_word_pkl
sentwordrawdic = wiki_data.sent_word_raw_pkl
postagdic = wiki_data.pos_tag_pkl

logging.info("Reading document vector model file...")

# Read document, sentence and word vector
with open(DOCVEC_PATH, 'r') as f:
    ss = f.readline().split()
    n_doc, n_units = int(ss[0]), int(ss[1])
    doc2index = {}
    index2doc = {}
    wdoc = np.empty((n_doc, n_units), dtype=np.float32)
    for i, line in enumerate(f):
        ss = line.split(',')
        assert len(ss) == n_units + 1
        doc = ss[0]
        doc2index[doc] = i
        index2doc[i] = doc
        wdoc[i] = np.array([float(s) for s in ss[1:]], dtype=np.float32)

logging.info("Reading sentence vector model file...")

with open(SENTVEC_PATH, 'r') as f:
    ss = f.readline().split()
    n_sent, n_units = int(ss[0]), int(ss[1])
    sent2index = {}
    index2sent = {}
    wsent = np.empty((n_sent, n_units), dtype=np.float32)
    for i, line in enumerate(f):
        ss = line.split(',')
        assert len(ss) == n_units + 1
        sent = ss[0]
        sent2index[sent] = i
        index2sent[i] = sent
        wsent[i] = np.array([float(s) for s in ss[1:]], dtype=np.float32)

logging.info("Reading word vector model file...")

with open(WORDVEC_PATH, 'r') as f:
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

sdoc = np.sqrt((wdoc * wdoc).sum(1))
wdoc /= sdoc.reshape((sdoc.shape[0], 1))

ssent = np.sqrt((wsent * wsent).sum(1))
wsent /= ssent.reshape((ssent.shape[0], 1))

sword = np.sqrt((wword * wword).sum(1))
wword /= sword.reshape((sword.shape[0], 1))

# For 'none' model, only generate gap-fill quizzes (not multiple-choice)
if args.model == 'none':
    if not os.path.isfile(INDEXTERM_PATH):
        logging.error("In case of \"none\" model, you need the index_term.pkl file")
        exit(0)
    else:
        logging.info("Reading index term...")
        index_term = pickle.load(open(INDEXTERM_PATH, 'rb'))
#        index_term_lower = [t.decode('utf-8').lower() for t in index_term]
        index_term_lower = [t.lower() for t in index_term]
        index_term_lower_lemmatized = [lemmatizer.lemmatize(t) for t in index_term_lower]
        index_term_lower_lemmatized = [t for t in index_term_lower_lemmatized if t not in stopwords.words('english')]

    logging.info("Performing sentence search...")
    quiz_doc_sent_tuple = []
    for doc in docsentdic:
        doc_title_words = doc.replace('_', ' ').replace('/', ' ').split()
        doc_title_words_lower = [t.lower() for t in doc_title_words]
        doc_title_words_lower_lemmatized = [lemmatizer.lemmatize(t) for t in doc_title_words_lower]
        doc_title_words_lower_lemmatized = [t for t in doc_title_words_lower_lemmatized if t not in stopwords.words('english')]
        sents = docsentdic[doc]
        for sent in sents:
            sent_words = sentworddic[sent]
            sent_words_lemmatized = [lemmatizer.lemmatize(t) for t in sent_words]
            # Check sentence length
            words = word_tokenize(sentwordrawdic[sent])
            if len(words) < SENTENCE_MIN_LENGTH:
                continue
            if list(set(hand_stopwords).intersection(set(sent_words))):
                continue
            # Check whether index term or document title word is in the sentence
            if (not list(set(index_term_lower_lemmatized).intersection(set(sent_words_lemmatized)))) and (not list(set(doc_title_words_lower_lemmatized).intersection(set(sent_words_lemmatized)))):
                continue
            quiz_doc_sent_tuple.append((doc, sent))

    logging.info("Performing gap search...")

    quiz_doc_sent_gap_tuple = []
    quizzes = []

    with open(args.output + '.txt', 'w') as f:
        quiz_count = 0
        for (doc, sent) in quiz_doc_sent_tuple:
            if quiz_count == args.num_quiz:
                break
            pos_list = nltk.pos_tag(word_tokenize(sentwordrawdic[sent]))
            sent_words = sentworddic[sent]
            doc_title_words = doc.replace('_', ' ').replace('/', ' ').split()
            doc_title_words_lower = [t.lower() for t in doc_title_words]
            doc_title_words_lower_lemmatized = [lemmatizer.lemmatize(t) for t in doc_title_words_lower]
            selected_gaps = []
            for word in sent_words:
                word_lemmatized = lemmatizer.lemmatize(word)
                if word_lemmatized in selected_gaps:
                    continue
                # Check whether the word is in the index term or the document title
                if word_lemmatized not in doc_title_words_lower_lemmatized and word_lemmatized not in index_term_lower_lemmatized:
                    continue
                # Check POS tag of the word
                gap_pos = [pos for (w, pos) in pos_list if w.lower() == word][0]
                if (not gap_pos.startswith('N')) and (not gap_pos.startswith('CD')):
                    continue
                gap = word
                quiz_doc_sent_gap_tuple.append((doc, sent, gap))
                selected_gaps.append(word_lemmatized)

                f.write(doc + '\n')
                print(doc)
                sentence = word_tokenize(sentwordrawdic[sent])
                replaced_sentence = []
                for j, w in enumerate(sentence):
                    if lemmatizer.lemmatize(gap) == lemmatizer.lemmatize(w.lower()):
                        replaced_sentence.append('______')
                    else:
                        replaced_sentence.append(w)
                f.write(' '.join(replaced_sentence) + '\n')
                print(' '.join(replaced_sentence))
                f.write(gap + '\n')
                print(gap)
                quizzes.append(Quiz(doc, ' '.join(replaced_sentence), gap, []))
                quiz_count += 1

        pickle.dump(quizzes, open(args.output + '.pkl', 'wb'))

elif args.model in ['joint', 'avg']:
    quiz_doc_sent_tuple = []

    # Select sentences for each document
    quiz_count = 0

    logging.info("Performing sentence selection...")

    for i in range(n_doc):
        vdoc = wdoc[i]
        sim = (wsent - vdoc)**2
        sim = np.sum(sim, axis=1)
        sim = np.sqrt(sim)
        count = 0
        for j in sim.argsort():
            if np.isnan(sim[j]):
                continue
            if index2sent[j] == '#PAD_SENT#':
                continue
            if index2sent[j] not in docsentdic[index2doc[i]]:
                continue
            words = word_tokenize(sentwordrawdic[index2sent[j]], lower=True)
            if len(words) < SENTENCE_MIN_LENGTH:
                continue
            if list(set(hand_stopwords).intersection(set(words))):
                continue
            quiz_doc_sent_tuple.append((i, j, float(sim[j])))
            count += 1
            quiz_count += 1
            if count == args.num_sent:
                break

    logging.info("Performing gap selection...")

    quiz_doc_sent_gap_tuple = []
    sim_doc_sent_gap = []

    # Select gaps for each sentence
    for (doc_idx, sent_idx, sim_ds) in quiz_doc_sent_tuple:
        vsent = wsent[sent_idx]
        sim = (wword - vsent) ** 2
        sim = np.sum(sim, axis=1)
        sim = np.sqrt(sim)
        pos_list = nltk.pos_tag(word_tokenize(sentwordrawdic[index2sent[sent_idx]]))
        count = 0
        for j in sim.argsort():
            if np.isnan(sim[j]):
                continue
            if index2word[j] == '#PAD_WORD#' or index2word[j] == '#EOS#':
                continue
            if index2word[j] not in sentworddic[index2sent[sent_idx]]:
                continue
            gap_pos = [pos for (word, pos) in pos_list if word.lower() == index2word[j]][0]
            if (not gap_pos.startswith('N')) and (not gap_pos.startswith('CD')):
                continue
            quiz_doc_sent_gap_tuple.append((doc_idx, sent_idx, j, sim_ds, float(sim[j])))
            count += 1
            if count == args.num_gap:
                break

    logging.info("Performing distractor selection...")

    quiz_doc_sent_gap_dist_tuple = []
    quizzes = []

    # Select distractor for each quiz
    with open(args.output + '.txt', 'w') as f:
        quiz_count = 0
        for (doc_idx, sent_idx, gap_idx, sim_ds, sim_sg) in quiz_doc_sent_gap_tuple:
            if quiz_count == args.num_quiz:
                break
            vgap = wword[gap_idx]
            sim = (wword - vgap) ** 2
            sim = np.sum(sim, axis=1)
            sim = np.sqrt(sim)
            pos_list = nltk.pos_tag(word_tokenize(sentwordrawdic[index2sent[sent_idx]]))
            gap_pos = [pos for (word, pos) in pos_list if word.lower() == index2word[gap_idx]][0]
            synonyms = wn.get_synonyms(index2word[gap_idx])
            same_lexname_words = wn.get_same_lexname_words(index2word[gap_idx])
            distractors = []
            sim_gd = []
            for i in sim.argsort():
                if np.isnan(sim[i]):
                    continue
                if index2word[i] == '#PAD_WORD#' or index2word[i] == '#EOS#':
                    continue
                if gap_idx == i:
                    continue
                if index2word[i].lower() in hand_stopwords:
                    continue
                # Exclude the words that belong to the same sentence as the gap
                if index2word[i] in sentworddic[index2sent[sent_idx]]:
                    continue
                # Compare POS of the gap and the word
                word_pos_list = postagdic[index2word[i]]
                if gap_pos not in word_pos_list:
                    continue
                # Check whether the gap and the word are synonyms
                if lemmatizer.lemmatize(index2word[i]) in synonyms:
                    continue
                # Compare WordNet lexical category (lexicographer file name, a.k.a lexname) of the gap and the word
                if wn.is_in_wordnet(lemmatizer.lemmatize(index2word[i])):
                    if lemmatizer.lemmatize(index2word[i]) not in same_lexname_words:
                        continue
                distractors.append(i)
                sim_gd.append(float(sim[i]))
                if len(distractors) == args.num_dist:
                    break

            # Calculate score
            quiz_doc_sent_gap_dist_tuple.append((doc_idx, sent_idx, gap_idx, distractors, sim_ds, sim_sg, sim_gd))
            score = 1 - ((1 / 6) * sim_ds + (1 / 6) * sim_sg + (1 / 6) * mean(sim_gd))
            print(score)

            # Write quizes to the file
            f.write(index2doc[doc_idx] + '\n')
            print(index2doc[doc_idx])
            sentence = word_tokenize(sentwordrawdic[index2sent[sent_idx]])
            replaced_sentence = []
            for j, w in enumerate(sentence):
                if lemmatizer.lemmatize(index2word[gap_idx]) == lemmatizer.lemmatize(w.lower()):
                    replaced_sentence.append('______')
                else:
                    replaced_sentence.append(w)
            f.write(' '.join(replaced_sentence) + '\n')
            print(' '.join(replaced_sentence))
            choices = [index2word[gap_idx]] + [index2word[i] for i in distractors]
            if CHOICES_SHUFFLE:
                random.shuffle(choices)
            f.write(' '.join(choices) + '\n')
            print(' '.join(choices))
            quizzes.append(Quiz(index2doc[doc_idx], ' '.join(replaced_sentence), index2word[gap_idx], [index2word[i] for i in distractors], score=score))
            quiz_count += 1

        pickle.dump(quizzes, open(args.output + '.pkl', 'wb'))
