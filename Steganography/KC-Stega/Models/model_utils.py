import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import json
import codecs
import math
from Config import config

def load_data(input_file):
    data_list = []
    with codecs.open(input_file, 'r', encoding='UTF-8') as f:
        for line in f:
            line_data = json.loads(line.encode('utf-8'))
            data_list.append(line_data)
    return data_list

def extraction(groups, file):
    #groups = groups.tolist()
        # glens = glens.tolist()
    total = []
    #groups是三维矩阵
    len_file = min(len(groups), config.test_batch_size)#如果遇到更小的batch_size只能一个batch一个batch地infer
    for i in range(len_file):
        ext = []
        for j in range(len(groups[0])):
            li = []
            for k in range(len(groups[0][0])):
                if groups[i][j][k]!=0:
                    li.append(file[i]['feature'][groups[i][j][k]-1])
            ext.append(li)
        total.append(ext)
    return total

def to_testfile(test_li):
    out = []
    for i in range(len(test_li)):
        out.append(give_topic(test_li[i]))

    file = open(config.test_file, 'w')
    for j in out:
        json_i = json.dumps(j, ensure_ascii=False)
        file.write(json_i + '\n')
    file.close()

def give_topic(topiclist):
    term = {}
    term['feature'] = topiclist
    term['title'] = ''
    term['largeSrc'] = ''
    term['refSrc'] = ''
    term['desc'] = ''
    term['file'] = ''
    term['专有属性'] = []
    term['共有属性'] = []
    term['segment'] = {}
    term['segment']['seg_0'] = {}
    term['segment']['seg_0']['segId'] = 0
    term['segment']['seg_0']['key_type'] = []
    term['segment']['seg_0']['order'] = []
    term['segment']['seg_0']['seg'] = ''
    return term

# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit*(2**i)
    return res

def get_bit_group(bpw, vocab_size):
    divide = math.pow(2, bpw)
    return [[int(divide*i+j) for i in range(int(vocab_size/divide))] for j in range(int(divide))]

def get_rnn_cell(rnn_type, num_layers, hidden_dim, keep_prob, scope):
    with tf.variable_scope(scope):
        lst = []
        for _ in range(num_layers):
            if rnn_type == 'gru':
                cell = tf.contrib.rnn.GRUCell(num_units=hidden_dim)
            else:
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            lst.append(cell)
        if num_layers > 1:
            res = tf.contrib.rnn.MultiRNNCell(lst)
        else:
            res = lst[0]
        return res

def rnn_state_shape(type, num_layers, shape):
    lst = []
    for _ in range(num_layers):
        if type == 'gru':
            lst.append(tf.TensorShape(shape))
        else:
            # lstm的 cell state ; hidden state两个需要初始化
            lst.append(tf.nn.rnn_cell.LSTMStateTuple(c=tf.TensorShape(shape), h=tf.TensorShape(shape)))
    if num_layers > 1:
        return tuple(lst)
    else:
        return lst[0]

def add_summary(summary_writer, global_step, tag, value):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)
    summary_writer.flush()

def restore_latest_model(model, best_checkpoint_dir, tmp_checkpoint_dir):
    tmp_latest_ckpt = tf.train.latest_checkpoint(tmp_checkpoint_dir)
    best_latest_ckpt = tf.train.latest_checkpoint(best_checkpoint_dir)
    if tmp_latest_ckpt is not None:
        tmp = eval(tmp_latest_ckpt.split("_")[1].split("-")[0])
    else:
        tmp = -1
    if best_latest_ckpt is not None:
        best = eval(best_latest_ckpt.split("_")[1].split("-")[0])
    else:
        best = -1
    start_epoch = tmp + 1 if tmp != -1 else 0
    model_dir = tmp_latest_ckpt
    if model_dir is None or (best != -1 and best > tmp):
        model_dir = best_latest_ckpt
        start_epoch = best + 1
    worse_step = tmp - best if best != -1 and tmp != -1 and tmp > best else 0
    if model_dir is not None:
        model.best_saver.restore(model.sess, model_dir)
    return start_epoch, worse_step, model

def restore_model(model, best_checkpoint_dir, tmp_checkpoint_dir):
    saver = model.best_saver
    latest_ckpt = tf.train.latest_checkpoint(best_checkpoint_dir)
    if latest_ckpt is None:
        saver = model.tmp_saver
        latest_ckpt = tf.train.latest_checkpoint(tmp_checkpoint_dir)
    if latest_ckpt is not None:
        saver.restore(model.sess, latest_ckpt)