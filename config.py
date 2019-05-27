import os


class Config():
    def __init__(self):
        for attr in self.__dir__():
            if attr.startswith('dir'):
                if not os.path.exists(self.__getattribute__(attr)):
                    os.makedirs(self.__getattribute__(attr))

    dir_data = 'data'
    dir_raw_data = os.path.join(dir_data, 'raw')
    dir_split_raw = os.path.join(dir_data, 'split_raw')

    path_word_list = os.path.join(dir_data, 'wordlist')
    path_class_list = os.path.join(dir_data, 'classlist')

    dir_save = 'data/checkpoints'
    path_best_model = os.path.join(dir_save, 'best.pth')

    print_freq = 50

    num_epoch = 30
    lr = 0.0004
    dropout = 0.5
    hidden_size = 300
    linear_size = 300
    embeds_dim = 100
