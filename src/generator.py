import preprocessing as pp
import numpy as np
import cv2
import string
from itertools import groupby
import unicodedata

class Tokenizer:
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=128):
        self.PAD_TK, self.UNK_TK = "¶", "¤"
        self.chars = (self.PAD_TK + self.UNK_TK + chars)

        self.PAD = self.chars.find(self.PAD_TK)
        self.UNK = self.chars.find(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""

        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
        text = " ".join(text.split())

        groups = ["".join(group) for _, group in groupby(text)]
        text = "".join([self.UNK_TK.join(list(x)) if len(x) > 1 else x for x in groups])
        encoded = []

        for item in text:
            index = self.chars.find(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def decode(self, text):
        """Decode vector to text"""

        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)
        decoded = pp.text_standardize(decoded)

        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")


class DataGenerator:

    def __init__(self, dataset, batch_size):
      self.index = dict()
      self.dataset = dataset
      self.batch_size = batch_size
      self.tokenizer = Tokenizer(string.printable[:95])


    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        self.index['train'] = 0

        while True:
            if self.index['train'] >= len(self.dataset['train']['dt']):
                self.index['train'] = 0

            ind = self.index['train']
            until = ind + self.batch_size
            self.index['train'] = until

            x_train = self.dataset['train']['dt'][ind:until]
            x_train = pp.normalization(x_train)
            
            y_train = [self.tokenizer.encode(y) for y in self.dataset['train']['gt'][ind:until]]
            y_train = [np.pad(y, (0, self.tokenizer.maxlen - len(y))) for y in y_train]
            y_train = np.asarray(y_train, dtype=np.int16)

            yield (x_train, y_train)


    def next_valid_batch(self):
        """Get the next batch from validation partition (yield)"""

        self.index['valid'] = 0

        while True:
            if self.index['valid'] >= len(self.dataset['valid']['dt']):
                self.index['valid'] = 0

            ind = self.index['valid']
            until = ind + self.batch_size
            self.index['valid'] = until

            x_valid = self.dataset['valid']['dt'][ind:until]
            x_valid = pp.normalization(x_valid)
            
            y_valid = [self.tokenizer.encode(y) for y in self.dataset['valid']['gt'][ind:until]]
            y_valid = [np.pad(y, (0, self.tokenizer.maxlen - len(y))) for y in y_valid]
            y_valid = np.asarray(y_valid, dtype=np.int16)

            yield (x_valid, y_valid)


    def next_test_batch(self):
        """Return model predict parameters"""

        self.index['test'] = 0

        while True:
            if self.index['test'] >= len(self.dataset['test']['dt']):
                self.index['test'] = 0
                break

            ind = self.index['test']
            until = ind + self.batch_size
            self.index['test'] = until

            x_test = self.dataset['test']['dt'][ind:until]
            x_test = pp.normalization(x_test)

            yield x_test