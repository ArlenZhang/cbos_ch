# coding:utf-8
import pickle
import numpy as np


UNK = "<UNK>"
PAD = "<PAD>"
PAD_LEN = 60


def load_vocab():
    with open("data/vocab.pickle", "rb") as vocab_fd:
        return pickle.load(vocab_fd)


if __name__ == '__main__':
    vocab = load_vocab()
    edus = []
    with open("data/gigaword.edus", "r", encoding='utf-8') as edu_fd:
        for line in edu_fd:
            edu = []
            for word in line.strip().split():
                if word == "<EDU_BREAK>" and edu:
                    edus.append(edu)
                    edu = []
                else:
                    edu.append(word)
            if edu:
                edus.append(edu)

    edu_ids = []
    for edu in edus:
        ids = []
        for word in edu:
            ids.append(vocab[word] if word in vocab else vocab[UNK])
        ids = ids[:PAD_LEN] if len(ids) >= PAD_LEN else ids + [vocab[PAD]] * (PAD_LEN - len(ids))
        edu_ids.append(ids)
    edu_ids = np.array(edu_ids, dtype=np.int32)
    np.save("data/edus.npy", edu_ids)
