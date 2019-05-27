import torch

from torch import nn, optim
from models.ESIM import ESIM
from preprocesser_v2 import MyDataset
from sklearn.metrics import classification_report, f1_score
from config import Config

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


class MyModel():
    def __init__(self, args):
        self.num_epoch = args.num_epoch
        self.lr = args.lr

        self.dir_save = args.dir_save
        self.path_best_model = args.path_best_model

        self.print_freq = args.print_freq

        self.dataset = MyDataset(dir_raw_data=args.dir_split_raw,
                                 path_word_list=args.path_word_list,
                                 path_class_list=args.path_class_list,
                                 tokenizer=str.split)

        self.model = ESIM(args,
                          num_words=len(self.dataset.TEXT.vocab.itos),
                          num_classes=len(self.dataset.LABEL.vocab.itos))

        self.model.to(device)

    def train(self):
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        max_dev_f1, max_test_f1 = 0, 0
        loss = 0

        for i, batch in enumerate(self.dataset.train_iter):
            present_epoch = self.dataset.train_iter.epoch

            if present_epoch == self.num_epoch:
                break

            pred = self.model(batch.sent1, batch.sent2)

            optimizer.zero_grad()
            batch_loss = loss_func(pred, batch.label)
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            if (i + 1) % self.print_freq == 0:
                prob, classes = torch.max(pred, -1)
                train_f1 = f1_score(batch.label.tolist(), classes.tolist(), average='macro')
                print('Training: in batch %d, loss %.6f, f1 %.6f' % (i + 1, batch_loss, train_f1))

            if (i + 1) % 500 == 0:
                dev_pred, dev_true, dev_loss = self.test(mode='dev')
                print('Dev: in batch %d, loss %.6f' % (i + 1, dev_loss))
                print(classification_report(dev_true, dev_pred,
                                            target_names=self.dataset.LABEL.vocab.itos))
                dev_f1 = f1_score(dev_true, dev_pred, average='macro')

                test_pred, test_true, test_loss = self.test(mode='test')
                print('Test: in batch %d, loss %.6f' % (i + 1, test_loss))
                print(classification_report(test_true, test_pred,
                                            target_names=self.dataset.LABEL.vocab.itos))
                test_f1 = f1_score(test_true, test_pred, average='macro')

                if dev_f1 > max_dev_f1:
                    max_dev_f1 = dev_f1
                    max_test_f1 = test_f1
                    print('保存模型。。')
                    torch.save({
                        'epoch': present_epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }, self.path_best_model)

                print('Best f1 score: dev %.6f, test %.6f' % (max_dev_f1, max_test_f1))

    def test(self, mode='test'):
        if mode == 'test':
            iterator = self.dataset.test_iter
        else:
            iterator = self.dataset.dev_iter

        loss_func = nn.CrossEntropyLoss()

        self.model.eval()
        all_pred = []
        all_true = []
        loss = 0

        for batch in iterator:
            pred = self.model(batch.sent1, batch.sent2)
            batch_loss = loss_func(pred, batch.label)

            _, pred = torch.max(pred, -1)

            all_pred.extend(pred.tolist())
            all_true.extend(batch.label.tolist())

            loss += batch_loss

        return all_pred, all_true, loss


if __name__ == '__main__':
    config = Config()
    model = MyModel(config)
    model.train()
