import os
import torch
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class CorpusGigaword(object):
    def __init__(self, path, threshold=2):
        """
        extends the above corpus class for English Gigaword
        :param path: path to data corpus directory
        """
        self.dictionary = Dictionary()
        self.tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
        self.translator = str.maketrans('', '', string.punctuation)
        self.stop_words = set(stopwords.words('english'))

        self.threshold = threshold
        self.end_token = '<end>'
        self.rare_token = '<rare>'
        self.counter = Counter()

        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def clean_doc(self, doc):
        words_only = self.tokenizer.tokenize(doc.strip())
        no_punctuation = [word.translate(self.translator) for word in words_only]
        lower_cased = [word.lower() for word in no_punctuation]

        self.counter.update(lower_cased)

        return ' '.join(lower_cased)

    def tokenize(self, filename):
        """
        - cleans docs
        - creates ditionary
        - tokenizes
        :param filename: data split name {train, valid, test}
        :return: token_ids mapping based on dictionary
        """
        assert os.path.exists(filename)
        with open(filename, 'r', encoding="utf8") as fp:
            docs = fp.readlines()
        fp.close()

        num_tokens = 0
        for idx in range(len(docs)):
            docs[idx] = self.clean_doc(docs[idx])
            words = docs[idx].split()
            num_tokens += len(words) + 1

        if 'test' not in filename:
            for doc in docs:
                words = doc.split()
                for word in words:
                    if self.counter.get(word) >= self.threshold:
                        self.dictionary.add_word(word)
        self.dictionary.add_word(self.end_token)
        self.dictionary.add_word(self.rare_token)

        token_ids = torch.Tensor(num_tokens).long()
        idx = 0
        for doc in docs:
            words = doc.split() + [self.end_token]
            for word in words:
                if word not in self.dictionary.word2idx:
                    token_ids[idx] = self.dictionary.word2idx[self.rare_token]
                else:
                    token_ids[idx] = self.dictionary.word2idx[word]
                idx += 1

        return token_ids
