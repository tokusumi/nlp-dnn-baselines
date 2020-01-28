from preprocess.mecab_tokenizer import (
    Tokenizer,
    Analyzer,
    RegexReplaceCharFilter,
    CompoundNounFilter,
    POSStopFilter
)


def test_mecab_analyzer():
    chars_filters = [RegexReplaceCharFilter(u'mecab', u'MeCab')]
    tokenizer = Tokenizer()
    token_filters = [CompoundNounFilter(), POSStopFilter(['記号', '助詞'])]
    analyzer = Analyzer(chars_filters, tokenizer, token_filters)

    res = analyzer.analyze('こんにちはmecab')
    assert [f.surface for f in res] == ['こんにちは', 'MeCab']

    res = analyzer.analyze('政治資金')
    assert [f.surface for f in res] == ['政治資金']

    res = analyzer.analyze('神〜')
    assert [f.surface for f in res] == ['神']
