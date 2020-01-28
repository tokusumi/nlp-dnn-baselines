
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tfdata_preprocess import get_test_dataset


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
    Test loss: 0.9957915084271491
    Test accuracy: 0.3856
    """
    train_data, test_data, params = get_test_dataset()
    max_len = params.get('max_len')
    num_words = params.get('vocab_size')
    batch_size = params.get('batch_size')
    num_classes = 3
    epochs = 3

    model = build_simple_dnn_model(max_len, num_words, num_classes)

    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()

    model.fit(train_data,
              epochs=epochs,
              validation_data=test_data)
    score = model.evaluate(test_data, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
