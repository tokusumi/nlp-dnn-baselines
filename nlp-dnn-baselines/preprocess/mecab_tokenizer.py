
import re
from MeCab import Tagger
import tensorflow as tf


def mecab_analyzer():
    """
    ref: 
    https://www.tensorflow.org/api_docs/python/tf/strings
    """
    # standarize texts
    char_filters = [RegexReplaceCharFilter(u'mecab', u'MeCab')]
    tokenizer = Tokenizer()
    token_filters = [CompoundNounFilter(), POSStopFilter(['記号', '助詞'])]
    analyze = Analyzer(char_filters, tokenizer, token_filters).analyze

    def _tokenizer(text, label):
        text = text.numpy().decode()
        tokenized_text = tf.strings.join([wakati.surface for wakati in analyze(text)], separator=' ')
        tokenized_text = tf.strings.lower(tokenized_text)
        return tokenized_text, label
    return _tokenizer


class Analyzer():
    def __init__(self, *args):
        self.args = [g for f in args for g in (f if hasattr(f, '__iter__') else [f])]

    def analyze(self, text):
        for _func in self.args:
            text = _func(text)
        return text


class Wakati():
    def __init__(self, ochasen):
        info = ochasen.split('\t')
        self.surface, self.kana, self.hira, tag, *_ = info
        self.tag, *_ = tag.split('-')


class Tokenizer():
    def __init__(self):
        self.tokenizer = Tagger("-Ochasen")

    def __call__(self, text):
        wakati = self.tokenizer.parse(text)
        wakati = [Wakati(f) for f in wakati.split('\n') if f != 'EOS' and f]
        return wakati


def RegexReplaceCharFilter(original, replace):
    def _filter(text):
        return re.sub(original, replace, text)
    return _filter


def CompoundNounFilter():
    def _filter(wakati):
        data = [wakati[0]]; data_append = data.append
        for past_index, words in enumerate(wakati[1:]):
            if words.tag == '名詞' and wakati[past_index].tag == '名詞':
                data[past_index].surface += words.surface
            else:
                data_append(words)
        return data
    return _filter


def POSStopFilter(pos):
    _pos = set(pos)

    def _filter(wakati):
        return [f for f in wakati if f.tag not in _pos]
    return _filter
