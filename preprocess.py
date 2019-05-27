import os
import nltk
import numpy as np

from collections import Counter
from config import Config


def filename(dir_):
    for _, _, files in os.walk(dir_):
        return files


def get_word2idx(path):
    with open(path, 'r', encoding='utf-8') as f:
        return {word: idx for idx, word in enumerate(f.readlines())}


def get_idx2word(path):
    with open(path, 'r', encoding='utf-8') as f:
        return {idx: word for idx, word in enumerate(f.readlines())}


def get_snli_data(dir_snli, dir_raw_data):
    files = filename(dir_snli)

    for file in files:
        if file[-4:] == '.txt' and file[:4] == 'snli':
            cleaned_data = []
            with open(os.path.join(dir_snli, file), 'r', encoding='utf-8') as f:
                f.readline()
                for line in f:
                    cleaned_line = [line.split('\t')[i] for i in (0, 5, 6)]
                    if cleaned_line[0] == '-':
                        continue
                    cleaned_data.append(cleaned_line)
            cleaned_path = os.path.join(dir_raw_data, file.split('_')[-1])
            with open(cleaned_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(['\t'.join(line) for line in cleaned_data]))


class Preprocessor(object):
    def __init__(self, config, bos=False, eos=False, min_number=3):
        self.dir_data = config.dir_data
        self.dir_raw = config.dir_raw_data
        self.dir_split_raw = config.dir_split_raw

        self.path_wordlist = config.path_word_list
        self.path_classlist = config.path_class_list

        self.bos = bos
        self.eos = eos
        self.min_number = min_number

    def split_raw_data(self):
        files = filename(self.dir_raw)
        for file in files:
            if file[-4:] != '.txt':
                continue
            line_list = []
            with open(os.path.join(self.dir_raw, file), 'r', encoding='utf-8') as f:
                for line in f:
                    content = line.strip().split('\t')
                    label = content[0]

                    sent1 = content[1].lower()
                    sent2 = content[2].lower()

                    sent1 = ' '.join(nltk.word_tokenize(sent1))
                    sent2 = ' '.join(nltk.word_tokenize(sent2))

                    line_list.append('\t'.join([label, sent1, sent2]))

            with open(os.path.join(self.dir_split_raw, file), 'w', encoding='utf-8') as f:
                f.write('\n'.join(line_list))

    def get_wordlist(self):
        text = []
        for file in filename(self.dir_split_raw):
            if not file.startswith('train'):
                continue

            with open(os.path.join(self.dir_split_raw, file), 'r', encoding='utf-8') as f:
                for line in f:
                    text.extend(line.strip().split())

        counter = Counter(text)

        wordlist = []
        if self.bos:
            wordlist.append('<BOS>')
        if self.eos:
            wordlist.append('<EOS>')
        wordlist.append('<OOV>')

        wordlist.extend([key for key in counter if counter[key] >= self.min_number])

        with open(self.path_wordlist, 'w', encoding='utf-8') as f:
            f.write('\n'.join(wordlist))

    def get_classlist(self):
        with open(os.path.join(self.dir_split_raw, 'train_label.txt'), 'r', encoding='utf-8') as f:
            classes = set([line.strip() for line in f])

        with open(self.path_classlist, 'w', encoding='utf-8') as f:
            f.write('\n'.join(list(classes)))

    def txt2npz(self):
        word2idx = get_word2idx(self.path_wordlist)
        class2idx = get_word2idx(self.path_classlist)




if __name__ == '__main__':
    config = Config()
    preprocessor = Preprocessor(config)
    preprocessor.split_raw_data()
