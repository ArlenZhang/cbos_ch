# coding:utf-8
import pickle
from collections import defaultdict

min_occur = 30
UNK = "<UNK>"
PAD = "<PAD>"


if __name__ == '__main__':
    freq_dict = defaultdict(int)
    with open("data/gigaword.edus", "r", encoding='utf-8') as edu_df:
        for line in edu_df:
            for word in line.strip().split():
                if word != '<EDU_BREAK>':
                    freq_dict[word] += 1

    vocab = {PAD: 0, UNK: 1}
    for word in sorted(freq_dict, key=freq_dict.get, reverse=True):
        if freq_dict[word] >= min_occur:
            vocab[word] = len(vocab)
    with open("data/vocab.pickle", "wb+") as vocab_fd:
        pickle.dump(vocab, vocab_fd)
