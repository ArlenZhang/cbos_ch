# coding:utf-8
import numpy as np
import pickle


class DataSet:
    def __init__(self, nepoch=4, nbatch=128, window=2, nneg=4):
        self.nepoch = nepoch
        self.nbatch = nbatch
        self.window = window
        self.nneg = nneg
        self.load_vocab()
        self.load_samples()

    def load_vocab(self):
        with open("data/vocab.pickle", "rb") as vocab_fd:
            self.vocab = pickle.load(vocab_fd)
            self.nvocab = len(self.vocab)

    def load_samples(self):
        self.samples = np.load("data/edus.npy")
        self.nsamples, self.seqlen = self.samples.shape

    def __iter__(self):
        self.epoch = 0
        self.batch = 0
        self.iter = 0

        for _ in range(self.nepoch):
            # sample indices of target edu
            indices = np.random.permutation(self.nsamples - self.window * 2 - 1) + self.window + 1
            ntargets = len(indices)
            self.epoch += 1
            self.batch = 0
            offset = 0
            while offset < ntargets:
                end = offset + self.nbatch if offset + self.nbatch <= ntargets else ntargets
                target_indices = indices[offset: end]
                context_indices = [list(range(i-self.window, i)) + list(range(i+1, i+1+self.window))
                                   for i in target_indices]
                negtive_indices = []
                for ti, cs in zip(target_indices, context_indices):
                    ns = []
                    for ni in range(self.nneg):
                        while True:
                            candidate = np.random.choice(indices, 1)[0]
                            if int(candidate) not in [ti] + cs + negtive_indices:
                                break
                        ns.append(candidate)
                    negtive_indices.append(ns)

                targets = np.take(self.samples, target_indices, 0)
                contexts = np.array([np.take(self.samples, i, 0) for i in context_indices])
                negtives = np.array([np.take(self.samples, i, 0) for i in negtive_indices])
                yield targets, contexts, negtives
                self.batch += 1
                self.iter += 1
                offset = end
