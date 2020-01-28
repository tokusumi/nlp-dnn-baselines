# # char-level CNN baseline
import tensorflow as tf
from tensorflow.keras.backend import tensorflow_backend
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Embedding
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Lambda, concatenate
from tensorflow.keras.callbacks import EarlyStopping


def Character_Level_CNN_layer(x, kernel_size, filter_size=256, dropout=0.5):
    x_conv = Conv1D(filter_size, kernel_size, padding="same", activation="relu")(x)

    def sum_pooling(x):
        return tf.math.reduce_sum(x, axis=1)

    x = Lambda(sum_pooling)(x_conv)
    x = Dropout(dropout)(x)
    return x


def make_model(maxlen: int, num_words: int, num_classes: int, activation="softmax", mingram=2, maxgram=5, dropout=0.5):
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

    x = Embedding(num_words, 32)(input_tensor)

    # convolution layers
    # size 1*mingram, ... 1*maxgram
    char_cnns = [Character_Level_CNN_layer(x, gram) for gram in range(mingram, maxgram)]

    # concatenate char-level cnn layers
    x = concatenate(char_cnns, axis=-1)

    # Fully Connected layers
    x = Dense(1024, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(dropout)(x)
    out = Dense(num_classes, activation=activation)(x)

    model = Model(input_tensor, out)

    return model


if __name__ == "__main__":

    '''
    Trains a char-level convnet on the imdb dataset.
    '''
    # const
    batch_size = 128
    num_classes = 1
    activation = "sigmoid"
    maxlen = 100
    num_words = 200
    epochs = 2

    x_train, x_test, y_train, y_test = make_dataset(maxlen, num_words, num_classes)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    model = make_model(
        maxlen=maxlen,
        num_words=num_words,
        num_classes=num_classes,
        activation=activation
    )
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
