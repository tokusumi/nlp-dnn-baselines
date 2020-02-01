
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

from preprocess.tfdata import get_test_dataset
from preprocess.utils import (
    load_pickle,
)


def build_simple_dnn_model(maxlen, num_words, num_classes, activation="softmax", dense_dim=128, num=2, dropout=0.2):
    model = Sequential()
    model.add(Dense(dense_dim, input_shape=(maxlen, ), activation='relu'))
    model.add(Dropout(dropout))
    for n in range(num):
        model.add(Dense(dense_dim, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(num_classes))
    model.add(Activation(activation))
    return model


if __name__ == "__main__":
    """
    simple DNN:
    ---
    Test loss: 0.9283508432062366
    Test accuracy: 0.5552
    """
    num_classes = 3
    epochs = 3

    vocab_set = load_pickle('./preprocess/vocab_set.pkl')
    # vocab_set = {}
    train_data, test_data, params, vocab_set = get_test_dataset(epochs=epochs,
                                                                vocab_set=vocab_set, max_len=30, cache_dir='cache/base')
    max_len = params.get('max_len')
    num_words = params.get('vocab_size')
    batch_size = params.get('batch_size')

    model = build_simple_dnn_model(max_len, num_words, num_classes)

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
