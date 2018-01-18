import re
import xml.etree.ElementTree as ET
import sys
from gensim import utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', default='book/database/book_xml_transformed/chapter/',
                    help='input xml files of chapters of database book')
parser.add_argument('--output', '-o', default='book/book.database.txt',
                    help='output file name')
args = parser.parse_args()

bold_tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}b"
color_tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}color"
p_tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
size_tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sz"
text_tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
ind_tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ind"
ilvl_tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ilvl"
pstyle_tag = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pStyle"
val_attrib = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
hanging_attrib = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hanging"

PAT_ALPHANUMERIC = re.compile('((\w)+)', re.UNICODE)

# For executing in pycharm
import nltk
nltk.data.path.append('/home/jhpark/datasets/')
from nltk import sent_tokenize


def remove_hyphen(text):
    text = text.replace("- ", "")
    return text


def get_db_book_chapter_name(p_node):
    cri1 = 0
    chapter_name = ""

    for child in p_node.iter():
        if child.tag == size_tag:
            if child.attrib[val_attrib] == '52':
                cri1 = 1

    is_chapter_name = (cri1 == 1)

    if is_chapter_name:
        for child in p_node.iter(text_tag):
            chapter_name += child.text

    return chapter_name


def get_db_book_section_name(p_node):
    cri1 = 0
    cri2 = 0
    section_name = ""

    for child in p_node.iter():
        if child.tag == color_tag:
            if child.attrib[val_attrib] == '00AEEF':
                cri1 = 1
        if child.tag == ind_tag:
            if hanging_attrib in child.attrib:
                if child.attrib[hanging_attrib] == '956':
                    cri2 = 1

    is_section_name = (cri1 + cri2 == 2)

    if is_section_name:
        for child in p_node.iter(text_tag):
            section_name += child.text

    return section_name


def get_db_book_subsection_name(p_node):
    cri1 = 0
    cri2 = 0
    cri3 = 0
    subsection_name = ""

    for child in p_node.iter():
        if child.tag == color_tag:
            if child.attrib[val_attrib] == '231F20':
                cri1 = 1
        if child.tag == bold_tag:
            cri2 = 1
        if child.tag == size_tag:
            if child.attrib[val_attrib] == '20':
                cri3 = 1

    is_subsection_name = (cri1 + cri2 + cri3 == 3)

    if is_subsection_name:
        for child in p_node.iter(text_tag):
            subsection_name += child.text

        pattern = re.compile(u"([0-9]+\.[0-9]+\.[0-9]+[ ]*)(.+)")
        m = pattern.match(subsection_name)

        if m:
            subsection_name = m.group(2)

    return subsection_name


def get_db_book_subsubsection_name(p_node):
    cri1 = 0
    cri2 = 0
    cri3 = 0
    cri4 = 0
    subsubsection_name = ""

    for child in p_node.iter():
        if child.tag == ilvl_tag:
            if child.attrib[val_attrib] == '3':
                cri1 = 1
        if child.tag == color_tag:
            if child.attrib[val_attrib] == '231F20':
                cri2 = 1
        if child.tag == pstyle_tag:
            if child.attrib[val_attrib] == "2":
                cri3 = 1
        if child.tag == bold_tag:
            cri4 = 1

    is_subsubsection_name = (cri1 + cri2 + cri3 + cri4 == 4)

    if is_subsubsection_name:
        for child in p_node.iter(text_tag):
            subsubsection_name += child.text

    return subsubsection_name


def get_db_book_text(p_node):
    text = ""
    is_curr_bold_tag = False
    is_curr_bold_word = False

    if get_db_book_chapter_name(p_node) or get_db_book_section_name(p_node) or get_db_book_subsection_name(
            p_node) or get_db_book_subsubsection_name(p_node):
        return text

    for child in p_node.iter():
        if child.tag == text_tag:
            text += child.text

    text = remove_hyphen(text)

    return text


def word_tokenize(content):
    return [
        token.encode('utf8') for token in tokenize(content, errors='ignore')
    ]


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


def process_db_book():
    f = open(args.output, 'r')

    curr_chapter_name = ""
    curr_section_name = ""
    curr_subsection_name = ""
    curr_subsubsection_name = ""
    curr_text = ""
    prev_doc_title = ""
    sent_num = 0

    paragraph_title = []

    with open(args.output, 'w') as f:
        for i in range(1, 31):
            print("Processing chapter %02d..." % i, file=sys.stderr)

            file_name = "chapter%02d.xml" % i
            file_path = args.input + file_name

            chapter = ET.parse(file_path)
            root = chapter.getroot()

            for p_node in root.iter(p_tag):
                this_chapter_name = get_db_book_chapter_name(p_node)
                this_section_name = get_db_book_section_name(p_node)
                this_subsection_name = get_db_book_subsection_name(p_node)
                this_subsubsection_name = get_db_book_subsubsection_name(p_node)
                this_text = get_db_book_text(p_node)

                if this_chapter_name:
                    paragraph_title = [this_chapter_name]
                    curr_chapter_name = this_chapter_name
                elif this_section_name:
                    paragraph_title = [curr_chapter_name, this_section_name]
                    curr_section_name = this_section_name
                elif this_subsection_name:
                    paragraph_title = [curr_chapter_name, curr_section_name, this_subsection_name]
                    curr_subsection_name = this_subsection_name
                elif this_subsubsection_name:
                    paragraph_title = [curr_chapter_name, curr_section_name, curr_subsection_name, this_subsubsection_name]
                    curr_subsubsection_name = this_subsubsection_name
                elif this_text:
                    this_doc_title = '/'.join(paragraph_title).replace(' ', '_').replace(',', '')
                    if prev_doc_title != this_doc_title:
                        if prev_doc_title != "":
                            f.write('END_OF_DOCUMENT\n')
                        f.write(this_doc_title + '\n')
                        sent_num = 0

                    sentences = sent_tokenize(this_text)
                    for sent in sentences:
                        words = word_tokenize(sent)
                        if len(words) < 5:
                            continue
                        f.write(this_doc_title + '_SENT%d\n' % sent_num)
                        f.write(' '.join([t.decode('utf-8') for t in words]) + '\n')
                        sent_num += 1
                    prev_doc_title = this_doc_title
        f.write('END_OF_DOCUMENT\n')

    print("Processing done!", file=sys.stderr)

process_db_book()
