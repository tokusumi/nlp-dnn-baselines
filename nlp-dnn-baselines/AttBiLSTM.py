import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, Bidirectional

from preprocess.tfdata import get_test_dataset
from preprocess.utils import (
    load_pickle,
)


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, emb_dim):
        super(AttentionLayer, self).__init__()
        self.emb_dim = emb_dim
        self.tanh = Activation('tanh')
        self.dense = Dense(1, use_bias=False, activation='softmax')
        self.activation = Activation('tanh')
        pass

    def __call__(self, x):
        # x.shape = (batch, sentence_length, emb_dim)
        t = self.tanh(x)
        t = self.dense(t)
        # t.shape = (batch, sentence_length, 1)
        out = self.activation(tf.matmul(x, t, transpose_a=True))
        out = tf.reshape(out, shape=(-1, self.emb_dim))
        return out


def build_model(maxlen, num_words, num_classes, activation="softmax", emb_dim=64, n_hidden=64, dropout=0.2):
    """
    Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification (2016)
    ref: https://www.aclweb.org/anthology/P16-2034/
    """
    model = Sequential()
    model.add(Embedding(
        num_words,
        emb_dim,
        input_length=maxlen
    ))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(
        n_hidden,
        recurrent_dropout=dropout,
        return_sequences=True,
    ), input_shape=(maxlen, emb_dim), merge_mode='sum'))

    model.add(AttentionLayer(emb_dim=n_hidden))

    model.add(Dense(num_classes, activation=activation))
    return model


if __name__ == "__main__":
    """
    train test Att-BiLSTM:
    ---
    Test loss: 0.27314918045001696
    Test accuracy: 0.8856
    """
    num_classes = 3
    epochs = 3

    vocab_set = load_pickle('./preprocess/vocab_set.pkl')
    # vocab_set = {}
    train_data, test_data, params, vocab_set = get_test_dataset(
        epochs=epochs, vocab_set=vocab_set, max_len=30, cache_dir='cache/base')
    max_len = params.get('max_len')
    num_words = params.get('vocab_size')
    batch_size = params.get('batch_size')

    model = build_model(max_len, num_words, num_classes)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    model.fit(train_data,
              epochs=epochs,
              validation_data=test_data)
    score = model.evaluate(test_data, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
