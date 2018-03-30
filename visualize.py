"""
    对训练得到的张量创建tensorflow projector
"""
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import torch
import jieba.posseg
import os
from torch.autograd import Variable
import pickle

visual_path = 'visualization'
tensor_path = 'data/wordemb10.pth'
visual_edus = "data/edu_choose.tsv"

vocab_path = "data/vocab.pickle"
word_emb = "data/wordemb10.pth"
voc_dict = None
voc_emb_dict = None

def load_data():
    global voc_dict, voc_emb_dict
    with open(vocab_path, "rb") as f:
        voc_dict = pickle.load(f)
    with open(word_emb, "rb") as f:
        voc_emb_dict = torch.load(f)
    print(type(voc_emb_dict))
    idx = Variable(torch.cuda.LongTensor([0]))
    print(idx)
    print(voc_emb_dict(idx))
    

# 对要处理的文件进行edbedding构建
# def data_process():
#     with open(visual_edus, "rb") as f:
#         for line in f:
#             # 对当前行的数据进行向量表示
#             temp_tuple_list = jieba.posseg.cut(line)
#             for word, pos in temp_tuple_list:
#                 vocab_words.add(word)
#                 vocab_pos.add(pos)


def visualize(num_visualize, embed_):
    """
        输入：要显示的EDU个数
        整体Tensor数据
    """
    most_common_words(visual_path, num_visualize)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 加载的embedding传入
        final_embed_matrix = embed_
        embedding_var = tf.Variable(final_embed_matrix[:num_visualize], name='embedding_e2v')
        sess.run(embedding_var.initializer)  # 获取数据，初始化变量
        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter(visual_path)
        # add embedding to the config file
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name  # 将指定数量的张良赋值给它
        # link this tensor to its metadata file, in this case the first NUM_VISUALIZE words of vocab
        embedding.metadata_path = 'vocab_' + str(num_visualize) + '.tsv'
        # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, os.path.join(visual_path, 'model_e2v.ckpt'))


def most_common_words(visual_fld, num_visualize):
    """
        create a list of num_visualize most frequent words to visualize on TensorBoard.
        saved to visualization/vocab_[num_visualize].tsv
    """
    words = open(os.path.join(visual_fld, 'vocab.tsv'), 'r').readlines()[:num_visualize]
    words = [word for word in words]
    file = open(os.path.join(visual_fld, 'vocab_' + str(num_visualize) + '.tsv'), 'w')
    for word in words:
        file.write(word)
    file.close()


if __name__ == "__main__":
    # tensor_data = torch.load(tensor_path)
    # visualize(100, tensor_data)
    load_data()
