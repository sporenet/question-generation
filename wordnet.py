from nltk.corpus import wordnet as wn


def get_synonyms(word):
    synonyms = []

    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name().lower())

    return list(set(synonyms))


def get_antonyms(word):
    antonyms = []

    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name().lower())

    return list(set(antonyms))


def get_same_lexname_words(word):
    lexnames = []
    same_lexname_words = []

    for synset in wn.synsets(word):
        lexnames.append(synset.lexname())

    for synset in wn.all_synsets():
        if synset.lexname() in lexnames:
            for lemma in synset.lemmas():
                same_lexname_words.append(lemma.name().lower())

    return list(set(same_lexname_words))


def is_in_wordnet(word_lemmatized):
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            if word_lemmatized == lemma.name().lower():
                return True
    return False
