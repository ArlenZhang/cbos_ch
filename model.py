# coding:utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable as Var


class CBOW(nn.Module):
    def __init__(self, vocab_size, hidden):
        nn.Module.__init__(self)
        self.wordemb = nn.Embedding(vocab_size, hidden)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, targets, contexts, negtives):
        targets_var = Var(torch.from_numpy(targets).long())
        negtives_var = Var(torch.from_numpy(negtives).long())
        inputs = torch.cat([targets_var.view(targets_var.size(0), 1, -1), negtives_var], 1)
        inputs_emb = self._emb(inputs)
        contexts_var = Var(torch.from_numpy(contexts).long())
        contexts_emb = self._emb(contexts_var)
        contexts_window = contexts_emb.size(-2)
        contexts_s = self._sent(contexts_emb).sum(-2) / contexts_window
        inputs_s = self._sent(inputs_emb)
        ps = torch.bmm(inputs_s, contexts_s.unsqueeze(-1)).squeeze(-1)
        labels = Var(torch.LongTensor([0] * ps.size(0)))
        loss = self.criterion(ps, labels)
        return loss

    def _emb(self, idx):
        mask = (idx > 0).unsqueeze(-1).float()
        inputs_shape = idx.size()
        seq_len = idx.size(-1)
        _idx = idx.view(-1, seq_len)
        emb = self.wordemb(_idx)
        return emb.view(*inputs_shape, -1) * mask

    def _sent(self, emb):
        return emb.sum(-2)
