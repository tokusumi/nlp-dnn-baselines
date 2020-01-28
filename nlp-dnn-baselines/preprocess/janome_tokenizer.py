from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import (
    RegexReplaceCharFilter
)
from janome.tokenfilter import (
    CompoundNounFilter,
    POSStopFilter,
    LowerCaseFilter
)
import tensorflow as tf


def janome_analyzer():
    """
    ref: 
    https://mocobeta.github.io/janome/api/janome.html#module-janome.tokenfilter
    """
    # standarize texts
    char_filters = [RegexReplaceCharFilter(u'蛇の目', u'janome')]
    tokenizer = Tokenizer()
    token_filters = [CompoundNounFilter(), POSStopFilter(['記号', '助詞']), LowerCaseFilter()]
    analyze = Analyzer(char_filters, tokenizer, token_filters).analyze

    def _tokenizer(text, label):
        tokenized_text = " ".join([wakati.surface for wakati in analyze(text.numpy().decode())])
        return tokenized_text, label
    return _tokenizer


def janome_analyzer_tf():
    """
    ref: 
    https://www.tensorflow.org/api_docs/python/tf/strings
    https://mocobeta.github.io/janome/api/janome.html#module-janome.tokenfilter
    """
    # standarize texts
    char_filters = [RegexReplaceCharFilter(u'蛇の目', u'janome')]
    tokenizer = Tokenizer()
    token_filters = [CompoundNounFilter(), POSStopFilter(['記号', '助詞'])]
    analyze = Analyzer(char_filters, tokenizer, token_filters).analyze

    def _tokenizer(text, label):
        text = text.numpy().decode()
        tokenized_text = tf.strings.join([wakati.surface for wakati in analyze(text)], separator=' ')
        tokenized_text = tf.strings.lower(tokenized_text)
        return tokenized_text, label
    return _tokenizer
