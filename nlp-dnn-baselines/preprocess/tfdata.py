import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Set, List
import os

from .custom_tokenizer import janome_analyzer_tf as ja_tokenizer


def get_test_dataset(epochs=1, vocab_set: Set[str] = {}, max_len=0, BATCH_SIZE=64, TEST_SIZE=5000, BUFFER_SIZE=50000, cache_dir='cache'):
    """
    main function. implement the followings,
    - download dataset in local disk (please replace them if you want to use different dataset)
    - get dataset (pipeline)

    return:
        train_data: tf.data.Dataset for train
        test_data: tf.data.Dataset for test
        params: the followings is included
            buffer_size [int]: buffer size
            test_size [int]: test data size
            batch_size [int]: batch size of dataset
            vocab_size [int]: vocabulry of tokens
            max_len [int]: maximux length of sequence. samples is padded to max_len
        vocab_set: set of string of vocabulary
    """
    # download dataset in local disk
    directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
    file_names = ['cowper.txt', 'derby.txt', 'butler.txt']
    file_paths = download_file(directory_url, file_names)

    # pipeline
    train_data, test_data, params, vocab_set = pipeline(
        file_names=file_names, file_paths=file_paths,
        vocab_set=vocab_set, max_len=max_len,
        BATCH_SIZE=BATCH_SIZE, TEST_SIZE=TEST_SIZE, BUFFER_SIZE=BUFFER_SIZE
    )

    return train_data, test_data, params, vocab_set


def download_file(directory_url: List[str], file_names: List[str]) -> List[str]:
    """download dataset in local disk"""
    file_paths = [
        tf.keras.utils.get_file(file_name, directory_url + file_name)
        for file_name in file_names
    ]
    return file_paths


def labeling_map_fn(file_names):
    """
    curried map function to load dataset from .txt files.
    note: this is curried function, which requires filenames and index of this are converted as label ID.
    usage:     
        >>> files = tf.data.Dataset.list_files(file_paths)
        >>> dataset = files.interleave(
            labeling_map_fn(file_names),
        )
    , which
        :file_pahts: List[str] = the list of full path of dataset files
        :file_names: List[str] = the list of file names of dataset files

    """
    def _get_label(dataset):
        """get filename from dataset contains fila_path, then convert it into label ID defined in filenames of arguments"""
        filename = dataset.numpy().decode().rsplit('/', 1)[-1]
        label = file_names.index(filename)
        return label

    def _labeler(example, label):
        """assign label in dataset"""
        return tf.cast(example, tf.string), tf.cast(label, tf.int64)

    def _labeling_map_fn(file_path: str):
        """main map function"""
        dataset = tf.data.TextLineDataset(file_path)
        label = tf.py_function(_get_label, inp=[file_path], Tout=tf.int64)
        labeled_dataset = dataset.map(lambda ex: _labeler(ex, label))
        return labeled_dataset
    return _labeling_map_fn


def load_dataset(file_paths: List[str], file_names: List[str], BUFFER_SIZE=1):
    """
    load sample data from three files.
    then, concatenate these data.
    then, shuffle it.
    """
    files = tf.data.Dataset.list_files(file_paths)

    dataset = files.interleave(
        labeling_map_fn(file_names),
    )
    all_labeled_data = dataset.shuffle(
        BUFFER_SIZE, reshuffle_each_iteration=False)

    return all_labeled_data


def tokenize_map_fn(tokenizer):
    """
    convert python function for tf.data map
    """
    def _tokenize_map_fn(text: str, label: int):
        return tf.py_function(tokenizer, inp=[text, label], Tout=(tf.string, tf.int64))
    return _tokenize_map_fn


def get_vocabulary(datasets) -> Set[str]:
    """create vocabulary with dataset. return set of vocabulary"""
    tokenizer = tfds.features.text.Tokenizer().tokenize

    def _tokenize_map_fn(text: str, label: int):
        def _tokenize(text: str, label):
            return tokenizer(text.numpy()), label
        return tf.py_function(_tokenize, inp=[text, label], Tout=(tf.string, tf.int64))

    dataset = datasets.map(
        _tokenize_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    vocab = {g.decode() for f, _ in dataset for g in f.numpy()}
    return vocab


def encoder(vocabulary_set: Set[str]):
    """encode text to numbers. must set vocabulary_set"""
    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set).encode

    def _encode(text: str, label: int):
        encoded_text = encoder(text.numpy())
        return encoded_text, label
    return _encode


def encode_map_fn(encoder):
    """convert python function for tf.data map"""
    def _encode_map_fn(text: str, label: int):
        return tf.py_function(encoder, inp=[text, label], Tout=(tf.int64, tf.int64))
    return _encode_map_fn


def split_train_test(data, TEST_SIZE: int, BUFFER_SIZE: int, SEED=123):
    """
    Split data into train and test data.
    Assuming input data has shuffled before
    args:
        TEST_SIZE: test data size
    """
    # because of reshuffle_each_iteration = True (default),
    # train_data is reshuffled if you reuse train_data.
    train_data = data.skip(TEST_SIZE).shuffle(BUFFER_SIZE, seed=SEED)
    test_data = data.take(TEST_SIZE)
    return train_data, test_data


def get_max_len(datasets) -> int:
    """calculate max sentence length from dataset"""
    tokenizer = tfds.features.text.Tokenizer().tokenize

    def _get_len_map_fn(text: str, label: int):
        def _get_len(text: str):
            return len(tokenizer(text.numpy()))
        return tf.py_function(_get_len, inp=[text, ], Tout=tf.int32)

    dataset = datasets.map(
        _get_len_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    max_len = max({f.numpy() for f in dataset})
    return max_len


def pipeline(file_names: List[str], file_paths: Set[str], vocab_set: Set[str] = {}, max_len=0, BATCH_SIZE=64, TEST_SIZE=5000, BUFFER_SIZE=50000, cache_dir='cache'):
    """
    main pipeline function. implement the followings,
    1. load dataset from file
    2. custom tokenize (normalize, replace, wakati, ...)
    3. create vocabulary
    4. calculate max sentence length
    5. encode text to numbers
    6. split dataset to train and test
    7. get batch and padding inputs
    8. settings for performance (cache and autotune)

    *skip 3/4 process if non-null vocab_set/max_len is passed.

    return: 
        train_data: tf.data.Dataset for train
        test_data: tf.data.Dataset for test
        params: the followings is included
            buffer_size [int]: buffer size
            test_size [int]: test data size
            batch_size [int]: batch size of dataset
            vocab_size [int]: vocabulry of tokens
            max_len [int]: maximux length of sequence. samples is padded to max_len
        vocab_set: set of string of vocabulary
    """
    params = {'buffer_size': BUFFER_SIZE, 'test_size': TEST_SIZE, 'batch_size': BATCH_SIZE}

    # create dataset loader
    datasets = load_dataset(file_paths, file_names, BUFFER_SIZE)

    # custom tokenizer
    datasets = datasets.map(tokenize_map_fn(ja_tokenizer()), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if not vocab_set:
        # create vocabulary
        vocab_set = get_vocabulary(datasets)

    if not max_len:
        # get max sentence length
        max_len = get_max_len(datasets)

    params["vocab_size"] = len(vocab_set) + 1
    params["max_len"] = max_len

    # encode text to numbers
    encoded_data = datasets.map(encode_map_fn(encoder(vocab_set)))

    # split train and test dataset
    train_data, test_data = split_train_test(encoded_data, TEST_SIZE, BUFFER_SIZE)

    # padding and get batch.
    train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([max_len], []), drop_remainder=True)
    test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([max_len], []), drop_remainder=False)

    # setting for performance. cache and prefetch
    DIR = os.path.dirname(os.path.abspath(__file__))
    train_data = train_data.cache(os.path.join(DIR, cache_dir, 'tftrain')).prefetch(tf.data.experimental.AUTOTUNE)
    test_data = test_data.cache(os.path.join(DIR, cache_dir, 'tftest')).prefetch(tf.data.experimental.AUTOTUNE)

    return train_data, test_data, params, vocab_set
