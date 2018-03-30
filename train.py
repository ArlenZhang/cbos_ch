# coding:utf-8
from dataset import DataSet
from model import CBOW
from torch import optim
import torch

HIDDEN = 100
LR = 0.001
LOG_EVERY = 10
EPOCH = 4
BATCH = 200
WINDOW = 2
NNEG = 4


if __name__ == '__main__':
    dataset = DataSet(nepoch=EPOCH, nbatch=BATCH, window=WINDOW, nneg=NNEG)
    model = CBOW(dataset.nvocab, 100)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for targets, contexts, negtives in dataset:
        optimizer.zero_grad()
        loss = model(targets, contexts, negtives)
        loss.backward()
        optimizer.step()
        if dataset.iter % LOG_EVERY == 0:
            print("[iter %-4d epoch %-2d batch %-3d]  loss %-.3f" % (dataset.iter, dataset.epoch, dataset.batch,
                                                                     loss.data[0]))
    torch.save(model.wordemb, 'data/wordemb.pth')
