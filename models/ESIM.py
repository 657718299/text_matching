import torch
from torch import nn
from torch.nn import functional


class ESIM(nn.Module):
    def __init__(self, args, num_words, num_classes):
        super(ESIM, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.hidden_size = args.hidden_size
        self.embeds_dim = args.embeds_dim
        self.num_words = num_words
        self.num_classes = num_classes
        self.embeds = nn.Embedding(self.num_words, self.embeds_dim)
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)

        self.lstm1 = nn.LSTM(self.embeds_dim,
                             self.hidden_size,
                             batch_first=True,
                             bidirectional=True)

        self.lstm2 = nn.LSTM(self.hidden_size * 8,
                             self.hidden_size,
                             batch_first=True,
                             bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(args.linear_size, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(args.linear_size, self.num_classes),
            nn.Softmax(dim=-1))

    def soft_attention_align(self, o1, o2, mask1, mask2):
        attention = torch.matmul(o1, o2.transpose(1, 2))
        mask1 = mask1.float().masked_fill(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill(mask2, float('-inf'))

        weight1 = functional.softmax(attention + mask2.unsqueeze(1), dim=-1)
        o1_align = torch.matmul(weight1, o2)
        weight2 = functional.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        o2_align = torch.matmul(weight2, o1)

        return o1_align, o2_align

    def submul(self, o1, o1_align):
        mul = o1 * o1_align
        sub = o1 - o1_align
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, q1):
        avg = functional.avg_pool1d(q1.transpose(1, 2), q1.size(1)).squeeze(-1)
        max_ = functional.avg_pool1d(q1.transpose(1, 2), q1.size(1)).squeeze(-1)
        return torch.cat([avg, max_], -1)

    def forward(self, *input):
        sent1, sent2 = input[0], input[1]
        mask1, mask2 = sent1.eq(1), sent2.eq(1)

        x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        o1_align, o2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        q1_combined = torch.cat([o1, o1_align, self.submul(o1, o1_align)], -1)
        q2_combined = torch.cat([o2, o2_align, self.submul(o2, o2_align)], -1)

        q1_composed, _ = self.lstm2(q1_combined)
        q2_composed, _ = self.lstm2(q2_combined)

        q1_rep = self.apply_multiple(q1_composed)
        q2_rep = self.apply_multiple(q2_composed)

        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity




