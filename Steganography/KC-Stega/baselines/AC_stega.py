#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: AC_stega.py
@author: ImKe at 2021/7/1
@email: thq415_ic@yeah.net
@feature: #Enter features here stegonography and bits extraction
"""
import torch
from model.mta import MTALSTM
from config import config
from utils import *
import argparse
import sys
import os
from dataset import Dataset
import random

torch.manual_seed(123)
random.seed(123)
torch.cuda.manual_seed_all(123)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Stega:
    def __init__(self, model, i2w, w2i, STRATEGY, precision):
        """
        class for steganography
        :param model: trained LM
        :param i2w: word id to word
        :param w2i: word to word id
        :param STRATEGY:
        :param precision:
        """
        super(Stega, self).__init__()
        self.model = model
        self.i2w = i2w
        self.w2i = w2i
        self.STRATEGY = STRATEGY
        self.use_gpu = config.use_gpu
        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.bit_stream = get_bitstream()
        self.precision = precision

    # Arithmetic Coding
    def predict_AC_kww(self, topics, num_chars, bit_num=1, topic_k=5, kww=True):
        """
        Bits hiding process regard to Arithmetic Coding steganography
        :param topics: topics as input constraint
        :param num_chars: max generation word count of one sentence
        :param bit_num: truncation number of candidate pool
        :param topic_k: k value for KeyWord-Wise decoder
        :param kww: True if use KeyWord-Wise decoder
        :return:
        """
        topic_k = topic_k
        bit_index = 0
        bit = bit_num
        output_idx = [self.w2i['<BOS>']]

        topics = [self.w2i[x] for x in topics]
        topics = torch.tensor(topics)
        topics = topics.reshape((1, topics.shape[0]))
        topics = topics.to(self.device)

        hidden = self.model.init_hidden(batch_size=1)
        # 初始化参数 coverage vector和attention权重
        coverage_vector = self.model.init_coverage_vector(topics.shape[0], topics.shape[1])
        attentions = torch.zeros(num_chars, topics.shape[1])
        stega_bit = ''
        max_val = 2 ** self.precision
        cur_interval = [0, max_val]
        output = torch.zeros(1, self.model.hidden_dim).to(self.device)
        X = torch.tensor(output_idx[-1]).view(1, 1).to(self.device)
        pred, output, hidden, attn_weight, coverage_vector = self.model.inference(inputs=X, topics=topics,
                                                                                  output=output,
                                                                                  hidden=hidden,
                                                                                  coverage_vector=coverage_vector,
                                                                                  seq_length=torch.tensor(50).reshape(1, 1).to(self.device))
        prob = torch.exp(pred)[-1, :].reshape(-1)
        prob[1] = 0
        prob = prob / prob.sum()
        indices = torch.multinomial(prob, 1)
        output_idx.append(int(indices[-1]))
        X = torch.cat([X, indices.view(1, 1)], 1)
        for t in range(1, num_chars):
            output = output.squeeze(0)
            pred, output, hidden, attn_weight, coverage_vector = self.model.inference(inputs=X, topics=topics, output=output,
                                                                                 hidden=hidden,
                                                                                 coverage_vector=coverage_vector,
                                                                                 seq_length=torch.tensor(50).reshape(1, 1).to(self.device))
            prob = torch.exp(pred)[-1, :].reshape(-1)
            prob[1] = 0
            prob = prob / prob.sum()
            _prob, _indices = prob.sort(descending=True)
            broadcast_equal = (topics.reshape(-1, 1) == _indices[:topic_k])
            cnt_matrix = broadcast_equal.nonzero()
            if cnt_matrix.sum() == 0 or kww is False:
                if self.STRATEGY == 'topk':
                    prob, indices = prob.sort(descending=True)
                    prob = prob[:2 ** bit]
                    indices = indices[:2 ** bit]
                #             print(prob, indices)
                elif self.STRATEGY == 'sample':
                    # indices = torch.multinomial(prob, 2 ** bit)
                    # prob = prob[indices]
                    noise_z = -torch.log(-torch.log(torch.rand(prob.shape))).to(self.device)
                    _, indices = torch.topk(torch.log(prob) + noise_z, 2 ** bit)
                    prob = prob[indices]
                elif self.STRATEGY == '':
                    prob, indices = prob.sort(descending=True)
                else:
                    raise Exception('no such strategy')

                cur_int_range = cur_interval[1] - cur_interval[0]  # 区间的大小  2^26
                cur_threshold = 1 / cur_int_range  # 每个区间多大
                if prob[-1] < cur_threshold:
                    k = max(2, (prob < cur_threshold).nonzero()[0].item())
                    prob = prob[:k]
                    indices = indices[:k]

                prob = prob / prob.sum()  # 截断后线性归一化
                prob = prob.double()
                prob *= cur_int_range  # 概率转换为多少个区间
                prob = prob.round().long()  # 四舍五入取整，区间数描述的概率

                cum_probs = prob.cumsum(0)  # 前面所有项的和的序列区间数描述的分布函数，按理讲最后应该与区间数相同
                overfill_index = (cum_probs > cur_int_range).nonzero()  # tensor([[299]])
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]  # [299] 去掉最后一个概率
                cum_probs += cur_int_range - cum_probs[-1]  # 分布函数加到和区间数相等，区间数表示的分布函数

                cum_probs += cur_interval[0]  # 分布函数的第一项从左区间开始

                message_bits = self.bit_stream[bit_index: bit_index + self.precision]  # 取了26位，但不是编码这26位，是用这26位锁定一个位置
                message_bits = [int(_) for _ in message_bits]
                message_idx = bits2int(reversed(message_bits))  # reverse只是为了计算int
                selection = (cum_probs > message_idx).nonzero()[0].item()  # 选择的单词的索引，int，选择第几个单词

                pred = indices[selection].view(1, 1)  # 一个数，代表选了哪个单词

                new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]  # 新的左区间 如果选了第一个单词（selection=0）就代表不需要动区间的左边界
                new_int_top = cum_probs[selection]

                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, self.precision)))  # 二进制的下边界
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, self.precision)))  # 二进制的上边界

                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)

                new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded  # 新二进制区间
                new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

                cur_interval[0] = bits2int(reversed(new_int_bottom_bits))  # 新的区间
                cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1  # +1 here because upper bound is exclusive



                stega_bit += self.bit_stream[bit_index:bit_index + num_bits_encoded]
                bit_index += num_bits_encoded
            else:
                pred = topics[0][cnt_matrix[:, 0][cnt_matrix[:, 1].argmin()]].view(1, 1)
                # print(self.i2w[int(pred[-1])])

            attentions[t] = attn_weight[0].data

            if int(pred[-1]) == self.w2i['<EOS>']:
                break
            X = torch.cat([X, pred], 1)
            output_idx.append(int(pred[-1]))
            # print(int(pred[-1]), self.i2w[int(pred[-1])])

        #     print("隐藏的比特数：{}".format(bit_index))
        return (''.join([self.i2w[i] for i in output_idx[1:]]), [self.i2w[i] for i in output_idx[1:]],
                attentions[:t + 1].t(), output_idx[1:], bit_index, stega_bit)

    def extract_AC_kkw(self, topics_id, stega_text, stega_bits, bit_num, kww=True):
        """
        Bits extraction process regard to Arithmetic Coding steganography
        :param topics: topics as input constraint
        :param stega_text: texts as stego
        :param stega_bits: secret bits embedded into the stego text
        :param bit_num: truncation number of candidate pool
        :param kww: True if use KeyWord-Wise decoder
        :return:
        """

        for index in range(len(stega_text)):
            topics = [self.w2i[x] for x in topics_id[index]]
            topics = torch.tensor(topics).to(self.device)
            topics = topics.reshape((1, topics.shape[0])) # size (1, 3)
            hidden = self.model.init_hidden(batch_size=1)
            stega_sentence = stega_text[index].strip()
            stega_bit = stega_bits[index].strip()
            decode_bit = ''
            start_word = stega_sentence.split()[0]
            x = torch.LongTensor([[self.w2i['<BOS>']]]).to(self.device)
            coverage_vector = self.model.init_coverage_vector(topics.shape[0], topics.shape[1])
            output = torch.zeros(1, self.model.hidden_dim).to(self.device)
            pred, output, hidden, attn_weight, coverage_vector = self.model.inference(inputs=x, topics=topics,
                                                                                      output=output,
                                                                                      hidden=hidden,
                                                                                      coverage_vector=coverage_vector,
                                                                                      seq_length=torch.tensor(50).reshape(1, 1).to(self.device))
            output = output.squeeze(0)
            x = torch.LongTensor([[self.w2i['<BOS>'], self.w2i[start_word]]]).to(self.device)
            max_val = 2 ** self.precision  # num of intervals
            cur_interval = [0, max_val]  # bottom inclusive, top exclusive
            for word in stega_sentence.split()[1:]:
                pred, output, hidden, attn_weight, coverage_vector = self.model.inference(inputs=x, topics=topics,
                                                                                          output=output,
                                                                                          hidden=hidden,
                                                                                          coverage_vector=coverage_vector,
                                                                                          seq_length=torch.tensor(50).reshape(1, 1).to(self.device))
                output = output.squeeze(0)
                prob = torch.exp(pred)[-1, :].reshape(-1)
                prob[1] = 0
                prob = prob / prob.sum()
                word = torch.tensor(self.w2i[word]).view(1,).to(self.device) #size (1,)
                # if key word: continue
                if (word == topics).sum() > 0 and kww is True:
                    x = torch.cat([x, torch.LongTensor([[word]]).to(self.device)], dim=1)
                    continue
                else:
                    if self.STRATEGY == 'topk':
                        prob, indices = prob.sort(descending=True)
                        prob = prob[:2 ** bit_num]
                        indices = indices[:2 ** bit_num]
                    elif self.STRATEGY == 'sample':
                        # indices = torch.multinomial(prob, 2 ** bit_num)
                        # prob = prob[indices]
                        noise_z = -torch.log(-torch.log(torch.rand(prob.shape))).to(self.device)
                        _, indices = torch.topk(torch.log(prob) + noise_z, 2 ** bit)
                        prob = prob[indices]
                    elif self.STRATEGY == '':
                        prob, indices = prob.sort(descending=True)
                    else:
                        raise Exception('no such strategy')

                    cur_int_range = cur_interval[1] - cur_interval[0]  # 区间的大小  2^26
                    cur_threshold = 1 / cur_int_range  # 每个区间多大
                    if prob[-1] < cur_threshold:
                        k = max(2, (prob < cur_threshold).nonzero()[0].item())
                        prob = prob[:k]
                        indices = indices[:k]

                    prob = prob / prob.sum()  # 截断后线性归一化
                    prob = prob.double()
                    prob *= cur_int_range  # 概率转换为多少个区间
                    prob = prob.round().long()  # 四舍五入取整，区间数描述的概率

                    cum_probs = prob.cumsum(0)  # 前面所有项的和的序列区间数描述的分布函数，按理讲最后应该与区间数相同
                    overfill_index = (cum_probs > cur_int_range).nonzero()
                    print("overfill", overfill_index)
                    if len(overfill_index) > 0:
                        cum_probs = cum_probs[:overfill_index[0]]  # [299] 去掉最后一个概率
                    cum_probs += cur_int_range - cum_probs[-1]  # 分布函数加到和区间数相等，区间数表示的分布函数

                    cum_probs += cur_interval[0]  # 分布函数的第一项从左区间开始

                    # print(indices, word)
                    selection = (indices == word).nonzero()[0].item() # indices size (32, )
                    # print(selection)
                    # print(cum_probs)

                    new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]  # 新的左区间 如果选了第一个单词（selection=0）就代表不需要动区间的左边界
                    new_int_top = cum_probs[selection]

                    new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, self.precision)))  # 二进制的下边界
                    new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, self.precision)))  # 二进制的上边界

                    num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)

                    # start decoding
                    new_bits = new_int_top_bits_inc[:num_bits_encoded]
                    decode_bit += ''.join([str(i) for i in new_bits])

                    new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded  # 新二进制区间
                    new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

                    cur_interval[0] = bits2int(reversed(new_int_bottom_bits))  # 新的区间
                    cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1  # +1 here because upper bound is exclusive

                    x = torch.cat([x, torch.LongTensor([[word]]).to(self.device)], dim=1)

            if decode_bit != stega_bit:
                print(stega_bit)
                print(stega_sentence)
                print(decode_bit)


        return decode_bit


    def generate(self, topics, method='beam_search', bit_num=1, kww=True, is_sample=True):
        """
        Generate sentences
        :param topics: topic constraint
        :param method: generation method AC for Arithmetic Coding /(UNfinished) HC for Huffman Coding
                       (UNfinished) ADG for ADaptive Grouping Coding / beam_search for beam search decoding
                       greedy for greedy decoding
        :param bit_num: truncation number of candidate pool for AC method
        :param kww: True if use KeyWord-Wise decoder for AC method
        :param is_sample: True if use categorical sampling for beam search & greedy method
        :return:
        """
        bits = 0
        stega_bits = []
        max_charnum = 100
        if method == 'AC':
            _, output_words, attentions, _, bits, stega_bits = self.predict_AC_kww(topics, max_charnum, bit_num=bit_num, kww=kww)
        elif method == 'HC':
            pass
        elif method == 'ADG':
            pass
        elif method=='beam_search':
            _, output_words, attentions, coverage_vector = self.model.beam_search(topics, max_charnum,  self.i2w, self.w2i, is_sample=is_sample)
            stega_bits = []
        elif method=='greedy':
            _, output_words, attentions, _ = self.model.predict_rnn(topics, max_charnum, self.i2w, self.w2i)
            stega_bits = []

        else:
            raise Exception(f'{method} generation is not implemented!')

        return (' '.join(output_words)), bits, stega_bits

    def extract(self, topics, stego_file, stega_bits, bit, method='AC', kww=True):
        """
        Extract secret bit for AC method
        :param topics: topic constraint
        :param stego_file:
        :param stega_bits:
        :param bit: rch for beam search decoding / greedy for greedy decoding
        :param method: AC for Arithmetic Coding / (UNfinished) HC for Huffman Coding / (UNfinished) ADG for ADaptive Grouping Coding
        :param kww: True if use KeyWord-Wise decoder
        :return:
        """
        if method == 'AC':
            decoded_bit = self.extract_AC_kkw(topics, stego_file, stega_bits, kww=kww, bit_num=bit)
        elif method == 'HC':
            pass
        elif method =='ADG':
            pass
        else:
            raise Exception(f'{method} extraction is not implemented')
        return decoded_bit

def get_args():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda x : x.lower() == 'true')
    parser.add_argument("--dataname", type=str, default='clothes')
    parser.add_argument("--Stra", type=str, default='sample')
    parser.add_argument("--method", type=str, default='AC')
    parser.add_argument("--precision", type=int, default=26)
    parser.add_argument("--max_bit", type=int, default=5)
    parser.add_argument("--sent_num", type=int, default=5)
    parser.add_argument("--kww", type=bool, default=True)
    parser.add_argument("--stega", type=bool, default=True)
    args = parser.parse_args(sys.argv[1:])
    return args

if __name__=="__main__":
    args = get_args()
    dataset = Dataset(args.dataname)
    save_folder = f"./ckpt/{args.dataname}"
    vocab = torch.load(f"{save_folder}/vocab.pkl")
    word_vec = torch.load(f"{save_folder}/word_vec.pkl")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MTALSTM(hidden_dim=config.hidden_dim, embed_dim=config.embedding_dim, num_keywords=config.num_keywords,
                    num_layers=config.num_layers, num_labels=len(vocab), weight=word_vec, vocab_size=len(vocab),
                    bidirectional=config.bidirectional)
    model.eval()
    load_ckpt_eval(50, save_folder, model, device)
    topics = load_data(f'./topics/{args.dataname}/3topic.json')
    w2i = {ch: i for i, ch in enumerate(vocab)}
    i2w = {i: ch for i, ch in enumerate(vocab)}
    Stega = Stega(model, i2w, w2i, args.Stra, args.precision)
    method = args.method
    result_dir = f'./results/{args.dataname}'
    os.makedirs(result_dir, exist_ok=True)
    with torch.no_grad():
        if args.stega:
            for bit in range(1, args.max_bit + 1):
                ts = []
                bit_number = []
                stega_bit = []
                for i, topic in enumerate(topics[:args.sent_num]):
                    ts0, bn0, stega_bit0 = Stega.generate(topics=topic, method='AC', bit_num=bit, kww=args.kww)
                    ts.append(ts0)
                    bit_number.append(bn0)
                    stega_bit.append(stega_bit0)
                    if (i + 1) % 500 == 0:
                        print(f'finish No.{i + 1} topic, {args.sen_num - i - 1} topics remaining.')
                print(f'Finish hiding in mta{args.method}-kww{args.kww}-{args.Stra}-{bit}bit.')
                with open(f'{result_dir}/mta{args.method}-kww{args.kww}-{args.Stra}-{bit}bit.txt', 'w') as f:
                    for i in ts:
                        f.write(i + '\n')
                with open(f'{result_dir}/mta{args.method}-kww{args.kww}-{args.Stra}-{bit}bit.bit', 'w') as f:
                    for i in stega_bit:
                        f.write(i + '\n')
        else:
            for bit in range(1, args.max_bit + 1):
                stego_file = []
                with open(f'{result_dir}/mta{args.method}-kww{args.kww}-{args.Stra}-{bit}bit.txt', 'r') as f:
                    stego_file = f.readlines()
                with open(f'{result_dir}/mta{args.method}-kww{args.kww}-{args.Stra}-{bit}bit.bit', 'r') as f:
                    stega_bits = f.readlines()
                decoded_bits = Stega.extract(topics[:args.sent_num], stego_file, stega_bits, method='AC', bit=bit, kww=args.kww)
                with open(f'{result_dir}/mta{args.method}-kww{args.kww}-{args.Stra}-{bit}bit-decode.bit', 'w') as f:
                    for i in decoded_bits:
                        f.write(i + '\n')
                print(f"Finish decoding of mta{args.method}-kww{args.kww}-{args.Stra}-{bit}bit.")
