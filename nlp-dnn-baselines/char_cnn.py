# # char-level CNN baseline
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Embedding
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate

from preprocess.tfdata import get_test_dataset
from preprocess.utils import (
    load_pickle,
    save_pickle
)


class CharlevelConvLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size,
                 filter_size=256,
                 dropout=0.5,
                 *args, **kwargs):
        super(CharlevelConvLayer, self).__init__()
        self.conv = Conv1D(filter_size,
                           kernel_size,
                           padding="same",
                           activation="relu")
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        x = self.conv(x)
        # sum pooling
        x = tf.math.reduce_sum(x, axis=1)
        x = self.dropout(x)
        return x


def build_model(maxlen: int, num_words: int, num_classes: int,
                activation="softmax", mingram=2, maxgram=5,
                emb_dim=32, filter_size=256,
                num_dense=3, dense_dim=1024,
                dropout=0.5):
    """
    char-level CNN architecture
    ref: https://arxiv.org/pdf/1702.08568.pdf

    arguments:
        maxlen: max sequence length
        num_words: vocabulary size
        num_classes: how many groups of sequence you want to classify
        activation: activation of last layer
        mingram: minimum gram number you want to extract feature from sequence
        maxgram: maximum gram number you want to extract feature from sequence

    input: 
        (batch, char-level index of sequence)
        ex) convert [["she is"], ..., ["he is"]] into [[5, 2, 3, 1, 4, 5], ..., [2, 3, 1, 4, 5, 0]]

    output: 
        Model instance
    """
    input_tensor = Input(batch_shape=(None, maxlen), dtype="int32")

    x = Embedding(num_words, emb_dim)(input_tensor)

    # convolution layers
    # size 1*mingram, ... 1*maxgram
    char_cnns = [CharlevelConvLayer(kernel_size=gram,
                                    filter_size=filter_size,
                                    dropout=dropout)(x)
                 for gram in range(mingram, maxgram + 1)]

    # concatenate char-level cnn layers
    x = concatenate(char_cnns, axis=-1)
    # => shape(Batch, (maxgram - mingram + 1)*filter_size)

    # Fully Connected layers
    for i in range(num_dense):
        x = Dense(dense_dim, activation="relu")(x)
        x = Dropout(dropout)(x)
    out = Dense(num_classes, activation=activation)(x)

    model = Model(input_tensor, out)
    return model


if __name__ == "__main__":

    '''
    Train test char-level convolution:
    ---
    Test loss: 0.6190159400052662
    Test accuracy: 0.6076
    '''
    # const
    num_classes = 3
    epochs = 3

    vocab_set = load_pickle('./preprocess/vocab_set_char.pkl')
    # vocab_set = {}
    train_data, test_data, params, vocab_set = get_test_dataset(
        epochs=epochs, vocab_set=vocab_set, max_len=62,
        char=True, cache_dir='cache/char')

    # save_pickle(vocab_set, './preprocess/vocab_set_char.pkl')

    hyperparams = {
        'maxlen': params.get('max_len'),
        'num_words': params.get('vocab_size'),
        'num_classes': num_classes,
        'activation': "softmax", 'mingram': 2, 'maxgram': 5,
        'emb_dim': 32, 'filter_size': 32,
        'num_dense': 2, 'dense_dim': 16,
        'dropout': 0.3
    }

    model = build_model(**hyperparams)
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
