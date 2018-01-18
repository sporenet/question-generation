import random
import re
import pickle
import numpy as np
from gensim import utils

PAT_ALPHANUMERIC = re.compile('((\w)+)', re.UNICODE)


def random_sample_quizzes(fname_in, fname_out, count):
    with open(fname_in, 'r') as f:
        for i, l in enumerate(f):
            pass
    quiz_count = (i + 1) / 3
    line_number = random.sample(range(0, int(quiz_count)), count)
    line_number = [i * 3 for i in line_number] + [i * 3 + 1 for i in line_number] + [i * 3 + 2 for i in line_number]

    with open(fname_in, 'r') as f, open(fname_out, 'w') as ff:
        for i, l in enumerate(f):
            if i in line_number:
                ff.write(l)


def select_topn_quizzes(fname_in, fname_out, n):
    quizzes_pkl = pickle.load(open(fname_in, 'rb'))
    quizzes_score = [q.score for q in quizzes_pkl]
    quizzes_score = np.array(quizzes_score)

    with open(fname_out, 'w') as f:
        for i in -quizzes_score.argsort()[:n]:
            quiz = quizzes_pkl[i]
            f.write(quiz.document + '\n')
            f.write(quiz.sentence + '\n')
            f.write(quiz.gap + ' ')
            f.write(' '.join(quiz.distractors) + '\n')


def tokenize(text, lowercase=False, deacc=False, errors="strict", to_lower=False, lower=False):
    """
    Iteratively yield tokens as unicode strings, removing accent marks
    and optionally lowercasing the unidoce string by assigning True
    to one of the parameters, lowercase, to_lower, or lower.
    Input text may be either unicode or utf8-encoded byte string.
    The tokens on output are maximal contiguous sequences of alphabetic
    characters (no digits!).
    list(tokenize('Nic nemůže letět rychlostí vyšší, než 300 tisíc kilometrů za sekundu!', deacc = True))
    [u'Nic', u'nemuze', u'letet', u'rychlosti', u'vyssi', u'nez', u'tisic', u'kilometru', u'za', u'sekundu']
    """
    lowercase = lowercase or to_lower or lower
    text = utils.to_unicode(text, errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = utils.deaccent(text)
    for match in PAT_ALPHANUMERIC.finditer(text):
        yield match.group()


def word_tokenize(content, lower=False):
    """
    Tokenize a sentence. The input string `content` is assumed
    to be mark-up free (see `filter_wiki()`).
    Return list of tokens as utf8 bytestrings. Ignore words shorted than 2 or longer
    that 15 characters (not bytes!).
    """
    # TODO maybe ignore tokens with non-latin characters? (no chinese, arabic, russian etc.)
    return [
        token for token in tokenize(content, errors='ignore', lowercase=lower)
    ]

if __name__ == "__main__":


    # select_topn_quizzes('quiz/quiz.book.database.joint.pkl', 'quiz/quiz.book.database.joint.top100.txt', 100)
    # select_topn_quizzes('quiz/quiz.book.database.joint.pkl', 'quiz/quiz.book.database.joint.top200.txt', 200)
    # select_topn_quizzes('quiz/quiz.book.database.joint.pkl', 'quiz/quiz.book.database.joint.top500.txt', 500)
    # select_topn_quizzes('quiz/quiz.book.biology.joint.pkl', 'quiz/quiz.book.biology.joint.top100.txt', 100)
    # select_topn_quizzes('quiz/quiz.book.biology.joint.pkl', 'quiz/quiz.book.biology.joint.top200.txt', 200)
    # select_topn_quizzes('quiz/quiz.book.biology.joint.pkl', 'quiz/quiz.book.biology.joint.top500.txt', 500)
    #
    # random_sample_quizzes('quiz/quiz.book.database.joint.top500.txt', 'quiz/quiz.book.database.joint.top500.sample50.txt', 50)
    # random_sample_quizzes('quiz/quiz.book.biology.joint.top500.txt',
    #                       'quiz/quiz.book.biology.joint.top500.sample50.txt', 50)
    random_sample_quizzes('quiz/quiz.book.database.joint.filter.0.3.txt', 'quiz/quiz.book.database.joint.filter.0.3.sample50.txt', 50)
    random_sample_quizzes('quiz/quiz.book.biology.joint.filter.0.3.txt',
                          'quiz/quiz.book.biology.joint.filter.0.3.sample50.txt', 50)
