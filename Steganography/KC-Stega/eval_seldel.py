#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@file:   main.py
@author: Haoqin
@date:   2020/08/

"""

import os
import sys
import re
import numpy as np
import json
import codecs
import argparse
from Config import config
import pickle
from one_gen import gen_sen
import time
import datetime
import utils


now = datetime.datetime.now()
global word_cnt
word_cnt = 0

def import_lib():
	global Dataset, tf, device_lib, PHVM, Dataset, model_utils
	import tensorflow as tf
	from tensorflow.python.client import device_lib

	import Dataset

	from Models import PHVM
	from Models import model_utils

def load_data(input_file):
    data_list = []
    with codecs.open(input_file, 'r', encoding='UTF-8') as f:
        for line in f:
            line_data = json.loads(line.encode('UTF-8'))
            data_list.append(line_data)
    return data_list


# gloabal parameters
"""
common = load_data('data/attribute_json_modified/common_attribute_0.5.json')
dress  = load_data('data/attribute_json_modified/dress_attribute_0.5.json')
pants  = load_data('data/attribute_json_modified/pants_attribute_0.5.json')
top    = load_data('data/attribute_json_modified/top_attribute_0.5.json')
"""
val_id = load_data('data/attribute_json_modified/val_id0.5.json')[0]
vocab  = pickle.load(open(config.vocab_file, "rb"))

"""
def regkeywords(pre_list, dic, sent):
    label_reg = []
    no_rep = []
    for i in dic[0].keys():
        for j in dic[0][i]:
            j = re.sub('/', '|', j)
            if re.match('.*?('+str(j)+')', sent) is not None:
                if '|' in j:
                    j = re.match('(.*?)\|', j).group(1)#特殊符号需要使用反斜线申明
                if j not in [pre_list[k][1] for k in range(len(pre_list))] and j not in no_rep:#不找重复的关键词
                    if '/' in i:
                        i = re.match('(.*?)/', i).group(1)
                    label_reg.append([i, j])
                    no_rep.append(j)
    return label_reg

def extract_topic(gen_sent):
    test_list = []
    for i in range(len(gen_sent)):
        tmp_list = regkeywords([], common, gen_sent[i][0]['desc'])
        tmp_list.extend(regkeywords(tmp_list, dress, gen_sent[i][0]['desc']))
        tmp_list.extend(regkeywords(tmp_list, pants, gen_sent[i][0]['desc']))
        tmp_list.extend(regkeywords(tmp_list, top, gen_sent[i][0]['desc']))
        test_list.append(tmp_list)
    return test_list
    
def delete_repwords(test_list):
    list_len = len(test_list)
    delete_list = []
    for i in range(list_len):
        for j in range(list_len):
            tmp = test_list[j][1]
            if (test_list[i][1] in tmp) and test_list[i][1]!=tmp:#寻找重复的描述词，但不可删除自己
                delete_list.append(test_list[i])
    for k in range(len(delete_list)):
        if delete_list[k] in test_list:
            test_list.remove(delete_list[k])
    return test_list
"""
def extract_topic_id(gen_id):
    extracted_id = []
    for sents in gen_id:
        tmp = set()
        for wid in sents:
            for val_key in val_id.keys():
                if wid in val_id[val_key]:
                    tmp.add(int(val_key))
        extracted_id.append(list(tmp))
    # print(extracted_id)
    return extracted_id

#将描述文字分词后用空格分开写入
def dump(texts, filename):
    file = open(filename, "w")
    for inst in texts:
        lst = []
        inst = " ".join(inst)
        lst.append({'desc': inst})
        file.write(json.dumps(lst, ensure_ascii=False) + "\n")
    file.close()

def cal_recall(label_list, gen_list):
    cnt = 0
    # label_list = [label_list[i][:-2] for i in range(len(label_list))]
    for i in range(len(label_list)):
        cnt_label = 0#计算每句话的重复二元组
        for j in range(len(label_list[i])):
            if label_list[i][j] in gen_list[i]:
                cnt_label+=1
        if cnt_label == len(label_list[i])-1:
            cnt+=1
    return cnt/len(label_list)

#更改！
def cal_precision(l0, l1):#l0是label，l1为抽取出来的
    cnt = 0
    ass_num = -2#数据中辅助生成二元组个数
    must_num = 1#必要二元组，但是无实际意义（遵循<类型，裤>类型）
    outclude = []#不准确表达二元组对应句子的编号
    # l0 = [l0[i][:ass_num] for i in range(len(l0))]
    for i in range(len(l0)):
        if len(l1[i]) == len(l0[i])- must_num:
            label_cnt = 0
            for j in l0[i][1:]:
                if j in l1[i]:
                    label_cnt+=1
            if label_cnt == len(l1[i]):
                cnt+=1
            else: outclude.append(i)
        else: outclude.append(i)
    return cnt/len(l1), outclude


def cal_target(test0_sent, test0_label, logger):
    # 抽取二元组
    test_list = extract_topic_id(test0_sent)
    # for i in range(len(test_list)):
    #     test_list[i] = delete_repwords(test_list[i])
    presicion1, missinglist = cal_precision(test0_label, test_list)
    logger.info('完整二元组recall：{}\n完整二元组precision：{} '.format(cal_recall(test0_label, test_list), presicion1))
    return missinglist, presicion1

def tojson(t, path):
    file = open(path, 'w')
    for i in t:
        json_i = json.dumps(i, ensure_ascii=False)
        file.write(json_i+'\n')
    file.close()


def extract_val_id(val_list):
    val_id = []
    for inst_num in range(len(val_list)):
        inst_val_list = []
        for tuple_num in range(len(val_list[inst_num])):
            inst_val_list.append(vocab.lookup(val_list[inst_num][tuple_num][1], 2))
        val_id.append(inst_val_list)
    return val_id

def cal_precision_stage(presicion_list):
    init = len(presicion_list)-1
    result = presicion_list[init] * (1-presicion_list[init-1]) + presicion_list[init-1]
    for k in range(len(presicion_list)-2, 0, -1):
        result = (1-presicion_list[k-1]) * result + presicion_list[k-1]
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda x : x.lower() == 'true')
    parser.add_argument("--input_topic", type=str, default='input_topic') # 手动输入标题的整体名称(默认在 data/topics 文件夹内)————文件夹内需要有input_topic1.jsonl文件为输入标题
    parser.add_argument("--gen_model", type=str, default='PHVM') # 生成句子的文件名（默认在result文件夹）
    parser.add_argument("--hide_strategy", type=str, default='RS') # strategy to encode secret bit (RS: Reject Sample; AC: Arithmatical Coding)
    parser.add_argument("--STRATEGY", type=str, default="sample")
    parser.add_argument("--resume", type=bool, default=False) # 生成句子的文件名（默认在result文件夹）
    # parser.add_argument("--test_file", type=str, default='test')#转移文件
    parser.add_argument("--gen_num", type=int, default=1)
    parser.add_argument("--gen_thresh", type=float, default=0.95)
    args = parser.parse_args(sys.argv[1:])
    return args


"""
# 挑选不合格二元组时使用
def main():
    sample_num = 300
    args = get_args()
    gen_sent = load_data(args.output_sentence)
    gui_label = load_data(args.input_topic)
    cnt0 = 0
    cnt1 = sample_num
    record_list = []
    #print(len(gen_sent))
    for i in range(int(len(gen_sent)/300)):
        gen_sent0 = gen_sent[cnt0: cnt1]
        gui_label0 = gui_label[cnt0: cnt1]
        cnt0+=sample_num
        cnt1+=sample_num
        test_list = extract_topic(gen_sent0)
        for j in range(len(test_list)):
            test_list[j] = delete_repwords(test_list[j])
        record_list.append([gui_label0[0] , cal_recall(gui_label0, test_list), cal_precision(gui_label0, test_list)])
    tojson(record_list, 'data/eval_metric.json')
"""


def gen_mul_time(args):
    brand_set = pickle.load(open(config.brand_set_file, "rb"))
    gen_thresh = args.gen_thresh
    os.makedirs("./logs", exist_ok=True)
    logger_name = f"{args.hide_strategy}_{now.month}.{now.day}_p{config.AC_precision}_t{config.truncated_vocab_size}_{args.STRATEGY}{config.bit_per_word}bit_{args.input_topic}"
    logger = utils.get_logger(logger_name)
    precision = [] # 每次单独生成句子的精确度
    missinglist = []
    bit_len = []
    text_id = []
    # 肯定得生成1次
    raw_label0 = load_data(config.data_dir + '/topics/{}.jsonl'.format(args.input_topic))  # 最开始输入的手动输入标签名
    _text_id, _bit_len = gen_sen(raw_label0, args.hide_strategy, args.STRATEGY) # 第一次生成一定需要手动输入的topic文件名称
    _text_id = np.squeeze(_text_id, 1).tolist()
    # print(_text_id)
    bit_len.append(_bit_len)
    text_id.append(_text_id)

    raw_label0_id = extract_val_id(raw_label0)
    # print(raw_label0_id)
    missinglist0, precision0 = cal_target(_text_id, raw_label0_id, logger)
    missinglist.append(missinglist0)
    precision.append(precision0)

    next_label = []
    next_label_id = []
    for i in missinglist0:
        next_label.append(raw_label0[i])
        next_label_id.append((raw_label0_id[i]))
    for _ in range(1, args.gen_num):
        _text_id, _bit_len = gen_sen(next_label, args.hide_strategy, args.STRATEGY) # 按照名字生成
        _text_id = list(np.squeeze(_text_id, 1))
        # print(_text_id)
        raw_label = next_label
        raw_label_id = next_label_id
        bit_len.append(_bit_len)
        text_id.append(_text_id)

        missinglist1, precision1 = cal_target(_text_id, next_label_id, logger)
        missinglist.append(missinglist1)
        precision.append(precision1)
        next_label = []
        next_label_id = []
        for j in missinglist1:
            next_label.append(raw_label[j])
            next_label_id.append(raw_label_id[j])
        if len(missinglist1) == 0 or precision1<=0.02:#当精确率达到了100% 或者单次生成准确率小于1%
            break
        if cal_precision_stage(precision) >= gen_thresh:#设置停止生成的标志
            break
    logger.info("============================================================================================================")
    logger.info('第{}次生成后的精度是{}\t'.format(1, precision[0])) # 计算第1次生成的精度
    if len(precision)>=2:#多次生成才会输出多次生成后的精度
        for jj in range(2, len(precision) + 1):
            logger.info('第{}次生成后的精度是{}\t'.format(jj, cal_precision_stage(precision[:jj])))  # 计算每次生成的精度
    #整合精确生成的句子到PHVM1.json文件中
        for k in range(len(missinglist)-1, 0, -1):#从倒数第2个list开始起
            gen_sen0 = text_id[k]
            gen_sen1 = text_id[k - 1]
            tmp = [] # 更新missinglist
            for i in range(len(gen_sen0)):
                if i not in missinglist[k]:
                    gen_sen1[missinglist[k-1][i]] = gen_sen0[i]
                    bit_len[k - 1][missinglist[k - 1][i]] = bit_len[k][i]
                    tmp.append(i)
            tmp.sort(reverse=True)#从底层抽取，否则会改变list的相对编号
            for i in tmp:
                missinglist[k - 1].remove(missinglist[k - 1][i])
    # print(gen_sen1)
    bpw_sum = 0
    # print(bit_len[0], text_id[0], len(text_id[0][0]), len(text_id[0][1]))
    for bit_n in range(len(bit_len[0])):
        bpw_sum += bit_len[0][bit_n] / len(text_id[0][bit_n])
    bpw_sum /= len(text_id[0])
    gen_sent = [[vocab.id2word[wid] if vocab.id2word[wid] not in brand_set else "BRAND" for wid in sents]for sents in text_id[0]]
    generated_filename = f'/{args.hide_strategy}_{now.month}.{now.day}_p{config.AC_precision}_t{config.truncated_vocab_size}_{args.STRATEGY}{config.bit_per_word}bit_{args.input_topic}.json'
    dump(gen_sent, config.result_dir + generated_filename)
    logger.info("============================================================================================================")
    logger.info("完成文本隐写")
    if args.hide_strategy == 'RS':
        final_bpw  = bpw_sum * config.bit_per_word
    elif args.hide_strategy == 'AC':
        final_bpw = bpw_sum
    elif args.hide_strategy == 'no':
        final_bpw = 0
    logger.info(f'经过{len(precision)}轮生成，最终结果保存在{generated_filename}文件中，真实的bpw为{final_bpw}')
"""
def gen_mul_time_0(gen_cnt):
    gen_thresh = 0.95
    args = get_args()
    precision = []#每次单独生成句子的精确度
    missinglist = []
    bit_len = []
    #肯定得生成1次
    _bit_len = gen_sen('{}1.jsonl'.format(args.input_topic), '{}1'.format(args.gen_model))#第一次生成一定需要手动输入的topic文件名称
    bit_len.append(_bit_len)
    # scores.shape [输入二元组组数，1，句子个数，词语个数，tgt_vocab_size]
    raw_label0 = load_data(config.data_dir + '/topics/{}1.jsonl'.format(args.input_topic))#最开始输入的手动输入标签名
    final_PHVM = load_data(config.result_dir + '/{}1.json'.format(args.gen_model))
    missinglist0, precision0 = cal_target(final_PHVM, raw_label0)
    missinglist.append(missinglist0)
    precision.append(precision0)

    next_label0 = []
    for i in missinglist0:
        next_label0.append(raw_label0[i])
    tojson(next_label0, config.data_dir + '/topics/{}2.jsonl'.format(args.input_topic))#第二次指导生成的二元组输入
    for ii in range(1, gen_cnt):
        _bit_len = gen_sen('{}{}.jsonl'.format(args.input_topic, ii+1), '{}{}'.format(args.gen_model, ii+1))#按照名字生成
        bit_len.append(_bit_len)
        raw_label = load_data(config.data_dir + '/topics/{}{}.jsonl'.format(args.input_topic, ii+1))
        missinglist1, precision1 = cal_target(load_data(config.result_dir + '/{}{}.json'.format(args.gen_model, ii+1)), raw_label)
        missinglist.append(missinglist1)
        precision.append(precision1)
        next_label = []
        for j in missinglist1:
            next_label.append(raw_label[j])
        if len(missinglist1) == 0 or precision1<=0.01: # 当精确率达到了100% 或者单次生成准确率小于1%
            break
        if cal_precision_stage(precision) >= gen_thresh:#设置停止生成的标志
            break
        tojson(next_label, config.data_dir + '/topics/{}{}.jsonl'.format(args.input_topic, ii+2))

    print('第{}次生成后的精度是{}\t'.format(1, precision[0])) # 计算第1次生成的精度
    if len(precision)>=2:#多次生成才会输出多次生成后的精度
        for jj in range(2, len(precision) + 1):
            print('第{}次生成后的精度是{}\t'.format(jj, cal_precision_stage(precision[:jj])))  # 计算每次生成的精度
    #整合精确生成的句子到PHVM1.json文件中
        for k in range(len(missinglist)-1, 0, -1):#从倒数第2个list开始起
            gen_sen0 = load_data(config.result_dir + '/{}{}.json'.format(args.gen_model, k+1))
            gen_sen1 = load_data(config.result_dir + '/{}{}.json'.format(args.gen_model, k))
            tmp = []#用来更新missinglist
            for i in range(len(gen_sen0)):
                if i not in missinglist[k]:
                    gen_sen1[missinglist[k-1][i]] = gen_sen0[i]
                    bit_len[k - 1][missinglist[k - 1][i]] = bit_len[k][i]
                    tmp.append(i)
            tmp.sort(reverse=True)#从底层抽取，否则会改变list的相对编号
            for i in tmp:
                missinglist[k - 1].remove(missinglist[k - 1][i])
            tojson(gen_sen1, config.result_dir + '/{}{}.json'.format(args.gen_model, k))
    print('完成{}轮生成，最终结果保存在{}1.json文件中'.format(len(precision), args.gen_model))
"""

if __name__ == '__main__':
    start = time.clock()
    args = get_args()
    gen_mul_time(args)
    elapsed = (time.clock() - start)
    print("Generation completed with {}s".format(elapsed))