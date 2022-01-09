#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: dataset.py
@author: ImKe at 2021/7/1
@email: thq415_ic@yeah.net
@feature: #Enter features here
"""
import numpy as np
import torch
from gensim.models import KeyedVectors
from random import shuffle
from tqdm import tqdm


class Dataset():
    def __init__(self, dataname):
        super(Dataset, self).__init__()
        self.dataname = dataname
        self.file_path = f'./data/{self.dataname}'

    def build_vocab(self):

        fvec = KeyedVectors.load_word2vec_format(f"{self.file_path}/corpus_mincount_1_305000_vec_original.txt", binary=False)
        word_vec = fvec.vectors
        vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        vocab.extend(list(fvec.vocab.keys()))
        word_vec = np.concatenate((np.array([[0]*word_vec.shape[1]] * 4), word_vec))
        word_vec = torch.tensor(word_vec).float()
        print("total %d words" % len(word_vec))
        vocab_check_point = f'{self.file_path}/vocab.pkl'
        word_vec_check_point = f'{self.file_path}/word_vec.pkl'
        torch.save(vocab, vocab_check_point)
        torch.save(word_vec, word_vec_check_point)

    def extract_sents(self, vocab):
        essays = []
        topics = []
        corpus_file = f"{self.file_path}/{self.dataname}.txt"
        num_lines = sum(1 for line in open(corpus_file, 'r'))
        with open(corpus_file) as f:
            for line in tqdm(f, total=num_lines):
                essay, topic = line.replace('\n', '').split(' </d> ')
                essays.append(essay.split(' '))
                topics.append(topic.split(' '))
            f.close()
        assert len(topics) == len(essays)
        tot_len = len(topics)
        word_to_idx = {ch: i for i, ch in enumerate(vocab)}
        idx_to_word = {i: ch for i, ch in enumerate(vocab)}
        corpus_indice = list(map(lambda x: [word_to_idx[w] if (w in word_to_idx) else word_to_idx['<UNK>'] for w in x], tqdm(essays[:int(tot_len * 0.8)])))
        topics_indice = list(map(lambda x: [word_to_idx[w] if (w in word_to_idx) else word_to_idx['<UNK>'] for w in x], tqdm(topics[:int(tot_len * 0.8)])))
        corpus_test = list(map(lambda x: [word_to_idx[w] if (w in word_to_idx) else word_to_idx['<UNK>'] for w in x], tqdm(essays[int(tot_len * 0.8):])))
        topics_test = list(map(lambda x: [word_to_idx[w] if (w in word_to_idx) else word_to_idx['<UNK>'] for w in x], tqdm(topics[int(tot_len * 0.8):])))
        return corpus_indice, topics_indice, corpus_test, topics_test, word_to_idx, idx_to_word

    def shuffleData(self, topics_indice, corpus_indice):
        ind_list = [i for i in range(len(topics_indice))]
        shuffle(ind_list)
        topics_indice = np.array(topics_indice)
        corpus_indice = np.array(corpus_indice)
        topics_indice = topics_indice[ind_list,]
        corpus_indice = corpus_indice[ind_list,]
        topics_indice = topics_indice.tolist()
        corpus_indice = corpus_indice.tolist()
        return topics_indice, corpus_indice

    def data_iterator(self, corpus_indice, topics_indice, batch_size, num_steps):
        epoch_size = len(corpus_indice) // batch_size
        for i in range(epoch_size):
            raw_data = corpus_indice[i*batch_size: (i+1)*batch_size]
            key_words = topics_indice[i*batch_size: (i+1)*batch_size]
            data = np.zeros((len(raw_data), num_steps+1), dtype=np.int64)
            for i in range(batch_size):
                doc = raw_data[i]
                tmp = [1]
                tmp.extend(doc)
                tmp.extend([2])
                tmp = np.array(tmp, dtype=np.int64)
                _size = tmp.shape[0]
                data[i][:_size] = tmp
            key_words = np.array(key_words, dtype=np.int64)
            x = data[:, 0:num_steps]
            y = data[:, 1:]
            mask = np.float32(x != 0)
            x = torch.tensor(x)
            y = torch.tensor(y)
            mask = torch.tensor(mask)
            key_words = torch.tensor(key_words)
            yield(x, y, mask, key_words)