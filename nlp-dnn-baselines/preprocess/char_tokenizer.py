import tensorflow as tf


def char_tokenizer():
    def _tokenizer(text, label):
        text = text.numpy().decode()
        tokenized_text = tf.strings.join([char for char in text], separator=' ')
        tokenized_text = tf.strings.lower(tokenized_text)
        return tokenized_text, label
    return _tokenizer
