import os

from torchtext import data
from nltk import word_tokenize
from collections import defaultdict
from torchtext.vocab import Vocab


def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


class MyDataset(object):
    def __init__(self, dir_raw_data, path_word_list, path_class_list, tokenizer=word_tokenize):
        self.TEXT = data.Field(tokenize=tokenizer, lower=True, batch_first=True)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        print('加载语料中。。。')
        self.train, self.dev, self.test = data.TabularDataset.splits(
            path=dir_raw_data,
            train='train.txt',
            validation='dev.txt',
            test='test.txt',
            format='tsv',
            fields=[('label', self.LABEL),
                    ('sent1', self.TEXT),
                    ('sent2', self.TEXT)]
        )

        print('构建词表中。。。')

        self.TEXT.build_vocab(self.train, self.dev, self.test)
        self.LABEL.build_vocab(self.train)

        with open(path_word_list, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.TEXT.vocab.itos))

        with open(path_class_list, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.LABEL.vocab.itos))

        self.train_iter, self.dev_iter, self.test_iter = data.Iterator.splits(
            datasets=(self.train, self.dev, self.test),
            sort_key=lambda x: data.interleave_keys(len(x.sent1), len(x.sent2)),
            batch_sizes=(32, 256, 256),
            device=-1
        )


