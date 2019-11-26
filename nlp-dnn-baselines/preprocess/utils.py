import pickle


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_pickle(vocab, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)
