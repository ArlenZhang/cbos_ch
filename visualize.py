"""
    对训练得到的张量创建tensorflow projector
"""
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import torch
import jieba
import os
import numpy as np
from torch.autograd import Variable
import pickle

visual_path = 'visualization'
tensor_path = 'data/wordemb10.pth'
visual_edus = "data/edu_choose.tsv"

vocab_path = "data/vocab.pickle"
word_emb = "data/wordemb10.pth"
voc_dict = None
voc_emb_dict = None

# edu最长
len_edu = 0
len_vec = 100
len_line = 0
# edu_embedding
edu_emb_matrix = None


def load_data():
    global len_edu, len_line
    global voc_dict, voc_emb_dict
    with open(vocab_path, "rb") as f:
        voc_dict = pickle.load(f)
    with open(word_emb, "rb") as f:
        voc_emb_dict = torch.load(f)
    # 加载edu文件，统计最长edu长度
    with open(visual_edus, "r") as f:
        for line in f:
            tmp_l = len(list(jieba.cut(line, cut_all=False)))
            if tmp_l > len_edu:
                len_edu = tmp_l
            len_line += 1


def get_tensor(idx):
    idx = Variable(torch.cuda.LongTensor([idx]))
    torch_vec = voc_emb_dict(idx)
    np_vec = torch_vec.cpu().data.numpy()
    # 转一般向量
    return np_vec


# 对要处理的文件过呢word_embedding数据
def data_process():
    global edu_emb_matrix
    with open(visual_edus, "r") as f:
        edu_emb_matrix = np.zeros(shape=(len_line, len_vec))
        edu_idx = 0
        for line in f:
            tmp_vec = np.zeros(shape=(1, len_vec))
            # 对当前行的数据进行向量表示
            temp_tuple_list = list(jieba.cut(line, cut_all=False))
            count_len = 0
            for word in temp_tuple_list:
                if word in voc_dict.keys():
                    tmp_vec = np.add(tmp_vec, get_tensor(voc_dict[word]))
                else:
                    tmp_vec = np.add(tmp_vec, get_tensor(voc_dict['<UNK>']))
                count_len += 1
                # 当前行中单词数不足
                while count_len < len_edu:
                    # padding 0 就不用操作了
                    count_len += 1
            # 对矩阵填充
            edu_emb_matrix[edu_idx] = tmp_vec
            edu_idx += 1


# len_line就是要展示的EDU个数
def visualize(num_visualize=len_line):
    """
        输入：要显示的EDU个数
        整体Tensor数据
    """
    most_common_words(num_visualize)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        embedding_var = tf.Variable(edu_emb_matrix[:num_visualize], name='embedding_e')
        sess.run(embedding_var.initializer)
        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter(visual_path)
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name  # 将指定数量的张量赋值给它
        embedding.metadata_path = 'edu_' + str(num_visualize) + '.tsv'
        # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, os.path.join(visual_path, 'model_e.ckpt'))


def most_common_words(num_visualize):
    """
        创建 展示数据
    """
    lines = open(os.path.join('data/edu_cdtb.tsv'), 'r').readlines()[:num_visualize]
    lines = [line for line in lines]
    with open(os.path.join('visualization/edu_' + str(num_visualize) + '.tsv'), 'w') as file:
        for line in lines:
            file.write(line.strip() + "\n")


if __name__ == "__main__":
    load_data()  # 加载师哥的数据
    data_process()  # 处理数据，得到真正需要的此向量
    visualize(100)  # 展示

""" 
    run tensorboard --logdir='visualization'
    http://ArlenIAC:6006
"""
