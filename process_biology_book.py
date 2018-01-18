import argparse
from utils import word_tokenize
from nltk import sent_tokenize

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', required=True,
                    help='input preprocessed text file')
parser.add_argument('--output', '-o', required=True,
                    help='output file name')
args = parser.parse_args()

title_text_pair = []

with open(args.input, 'r') as f:
    lines = f.read().splitlines()
    title = []
    text = []
    this_title = ''
    this_text = ''
    for line in lines:
        if len(word_tokenize(line)) < 30 and line[-1] != '.':
            this_title = line
        else:
            this_text = line
            title_text_pair.append((this_title, this_text))

with open(args.output, 'w') as f:
    titles = []
    for title, text in title_text_pair:
        title = title.replace(' ', '_')
        title = title.replace(',', '')
        if title in titles:
            title = title + '_'
        titles.append(title)
        f.write(title + '\n')
        sents = [s for s in sent_tokenize(text) if len(word_tokenize(s)) >= 5]
        for i, sent in enumerate(sents):
            f.write(title + '_SENT%d\n' % i)
            words = word_tokenize(sent, lower=False)
            f.write(' '.join(words) + '\n')
        f.write('END_OF_DOCUMENT\n')
