from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN

from preprocess.tfdata import get_test_dataset
from preprocess.utils import (
    load_pickle,
)


def build_simple_rnn_model(maxlen, num_words, num_classes, activation="softmax", emb_dim=128, n_hidden=200):
    model = Sequential()
    model.add(Embedding(
        num_words,
        emb_dim,
        input_length=maxlen)
    )

    model.add(SimpleRNN(
        n_hidden,
        batch_input_shape=(None, maxlen, emb_dim),
        dropout=0.2,
        recurrent_dropout=0.2,
        return_sequences=False)
    )
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation(activation))
    return model


if __name__ == "__main__":
    """
    train test simple RNN:
    ---
    Test loss: 0.48383093182044695
    Test accuracy: 0.79
    """
    num_classes = 3
    epochs = 3

    vocab_set = load_pickle('preprocess/vocab_set.pkl')
    # vocab_set = {}
    train_data, test_data, params, vocab_set = get_test_dataset(epochs=epochs, vocab_set=vocab_set, max_len=30)
    max_len = params.get('max_len')
    num_words = params.get('vocab_size')
    batch_size = params.get('batch_size')

    model = build_simple_rnn_model(max_len, num_words, num_classes)

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
