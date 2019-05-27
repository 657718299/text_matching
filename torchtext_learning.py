from nltk import word_tokenize
from torchtext import data


TEXT = data.Field(tokenize=word_tokenize, lower=True)
LABEL = data.Field(sequential=False, unk_token=None)
train, dev, test = data.TabularDataset.splits(path='./data/raw',
                                              train='train.txt',
                                              validation='dev.txt',
                                              test='test.txt',
                                              format='tsv',
                                              fields=[('label', LABEL),
                                                      ('sent1', TEXT),
                                                      ('sent2', TEXT)])

TEXT.build_vocab(train, test, dev)
LABEL.build_vocab(train)
vocab_dict = TEXT.vocab.stoi

train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train, dev, test), sort_key=lambda x: len(x.Text),
        batch_sizes=(32, 256, 256), device=-1, repeat=True)

