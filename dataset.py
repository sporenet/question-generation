import codecs
import math
import nltk

PAD_SENT = '#PAD_SENT#'
PAD_WORD = '#PAD_WORD#'
EOS = '#EOS#'


class WikiData:
    def __init__(self, file_name, ws, ww):
        self.file_name = file_name
        self.window_sent = ws
        self.window_word = ww

        self.doc_title = []
        self.sent_title = []
        self.doc_sent = []
        self.sent_word = []
        self.docs = {}
        self.sents = {}
        self.words = {}

        self.doc_sent_pkl = {}
        self.sent_word_pkl = {}
        self.sent_word_raw_pkl = {}
        self.doc_word_pkl = {}
        self.pos_tag_pkl = {}

        self.l_doc_sent = []
        self.l_sent_word = []

    def load_data(self):
        with codecs.open(self.file_name, 'r', 'utf-8') as f:
            curr_doc_title = True
            curr_sent_title = False

            self.words[PAD_WORD] = 0
            self.sents[PAD_SENT] = 1

            padding_word = [self.words[PAD_WORD]] * math.floor(self.window_word)
            padding_sent = [self.sents[PAD_SENT]] * math.floor(self.window_sent)

            count = 2
            num_of_sent_in_doc = 0

            #TODO: doc title이 같은게 있으면 에러, 따라서 미리 rename하도록!

            for line in f:
                if curr_doc_title:  # Encounters document title
                    doc = line.rstrip()
                    if doc not in self.docs:
                        self.docs[doc] = count
                        self.doc_sent_pkl[doc] = set()
                        self.doc_word_pkl[doc] = set()
                        count += 1
                        self.doc_title.append(self.docs[doc])
                    curr_doc_title = False
                    curr_sent_title = True
                    self.doc_sent += padding_sent
                    num_of_sent_in_doc = 0
                elif line.rstrip() == 'END_OF_DOCUMENT':  # Encounters end of document
                    self.doc_sent += padding_sent
                    self.l_doc_sent.append(num_of_sent_in_doc + 2 * len(padding_sent))
                    curr_doc_title = True
                    curr_sent_title = False
                    num_of_sent_in_doc = 0
                elif curr_sent_title:  # Encounters sentence title
                    sent = line.rstrip()
                    if sent not in self.sents:
                        self.sents[sent] = count
                        self.sent_word_pkl[sent] = set()
                        self.sent_word_raw_pkl[sent] = ''
                        count += 1
                        self.doc_sent_pkl[doc].add(sent)
                        self.sent_title.append(self.sents[sent])
                        self.doc_sent.append(self.sents[sent])
                    curr_doc_title = False
                    curr_sent_title = False
                    num_of_sent_in_doc += 1
                else:  # Encounters words
                    line = line.strip()
                    pos_tags = nltk.pos_tag(line.split(' '))
                    self.l_sent_word.append(len(pos_tags) + 2 * len(padding_word))
                    for i, (word, pos) in enumerate(pos_tags):
                        if word.lower() not in self.words:
                            self.pos_tag_pkl[word.lower()] = set()
                            self.words[word.lower()] = count
                            count += 1
                            self.doc_word_pkl[doc].add(word.lower())
                        self.sent_word_pkl[sent].add(word.lower())
                        self.sent_word_raw_pkl[sent] += word + " "
                        self.pos_tag_pkl[word.lower()].add(pos)
                    self.sent_word += padding_word + [self.words[word.lower()] for (word, pos) in pos_tags] + padding_word
                    self.sent_word_raw_pkl[sent] = self.sent_word_raw_pkl[sent].rstrip()
                    curr_doc_title = False
                    curr_sent_title = True


class WikiDataW2V:
    def __init__(self, file_name):
        self.file_name = file_name

        self.words = {}
        self.all_words = []

    def load_data(self):
        with codecs.open(self.file_name, 'r', 'utf-8') as f:
            curr_doc_title = True
            curr_sent_title = False

            self.words[EOS] = 0

            count = 1
            for line in f:
                if curr_doc_title:  # Encounters document title
                    curr_doc_title = False
                    curr_sent_title = True
                elif line.rstrip() == 'END_OF_DOCUMENT':  # Encounters end of document
                    curr_doc_title = True
                    curr_sent_title = False
                elif curr_sent_title:  # Encounters sentence title
                    curr_doc_title = False
                    curr_sent_title = False
                else:  # Encounters words
                    line = line.rstrip()
                    words = line.split(' ')
                    for w in words:
                        if w.lower() not in self.words:
                            self.words[w.lower()] = count
                            count += 1
                        self.all_words.append(self.words[w.lower()])
                    self.all_words.append(self.words[EOS])
                    curr_doc_title = False
                    curr_sent_title = True


class GloVeData:
    def __init__(self, file_name):
        self.file_name = file_name
        self.model = {}

    def load_data(self):
        with codecs.open(self.file_name, 'r', 'utf-8') as f:
            for line in f:
                splitline = line.split()
                word = splitline[0]
                embedding = [float(val) for val in splitline[1:]]
                self.model[word] = embedding

    def get_vector(self, word):
        try:
            return self.model[word]
        except KeyError:
            print("Embedding for %s not exists in GloVe" % word)


class Quiz:
    def __init__(self, document, sentence, gap, distractors, score=0.0):
        self.document = document
        self.sentence = sentence
        self.gap = gap
        self.distractors = distractors
        self.score = score