#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: mobile_mta.py
@author: ImKe at 2021/7/13
@email: thq415_ic@yeah.net
@feature: #Enter features here
"""

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import Parameter, LayerNorm
from torch.autograd import Variable
import torch.jit as jit
import torch.nn.functional as F
import time
import os
import math
from tqdm import tqdm
import collections
from collections import namedtuple
from config import config
from model.model_util import *

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


class MTALSTM(nn.Module):
    def __init__(self, hidden_dim, embed_dim, num_keywords, num_layers, weight, vocab_size,
                 num_labels, bidirectional, dropout=0.5, **kwargs):
        super(MTALSTM, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.device = 'cuda' if config.use_gpu else 'cpu'
        if num_layers <= 1:
            self.dropout = 0
        else:
            self.dropout = dropout
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.Uf = nn.Linear(embed_dim * num_keywords, num_keywords, bias=False)

        # attention decoder
        self.decoder = AttentionDecoder(hidden_size=hidden_dim,
                                        embed_size=embed_dim,
                                        num_layers=num_layers,
                                        dropout=dropout)

        # adaptive softmax
        self.adaptiveSoftmax = nn.AdaptiveLogSoftmaxWithLoss(hidden_dim,
                                                             num_labels,
                                                             cutoffs=[round(num_labels / 20),
                                                                      4 * round(num_labels / 20)])

    def forward(self, inputs, topics, output, hidden=None, mask=None, target=None, coverage_vector=None,
                seq_length=None):
        embeddings = self.embedding(inputs)
        topics_embed = self.embedding(topics)
        ''' calculate phi [batch x num_keywords] '''
        phi = None
        phi = torch.sum(mask, dim=1, keepdim=True) * torch.sigmoid(
            self.Uf(topics_embed.reshape(topics_embed.shape[0], -1).float()))

        # loop through sequence
        inputs = embeddings.permute([1, 0, 2]).unbind(0)
        output_states = []
        attn_weight = []
        for i in range(len(inputs)):
            output, hidden, score, coverage_vector = self.decoder(input=inputs[i].unsqueeze(0),
                                                                  output=output,
                                                                  hidden=hidden,
                                                                  phi=phi,
                                                                  topics=topics_embed,
                                                                  coverage_vector=coverage_vector)  # [seq_len x batch x embed_size]
            output_states += [output]
            attn_weight += [score]

        output_states = torch.stack(output_states)
        attn_weight = torch.stack(attn_weight)

        # calculate loss py adaptiveSoftmax
        outputs = self.adaptiveSoftmax(output_states.reshape(-1, output_states.shape[-1]), target.t().reshape((-1,)))

        return outputs, output_states, hidden, attn_weight, coverage_vector

    @torch.jit.export
    def inference(self, inputs, topics, output, hidden=None, mask=None, coverage_vector=None, seq_length=None):
        embeddings = self.embedding(inputs)
        topics_embed = self.embedding(topics)

        phi = None
        phi = seq_length.float() * torch.sigmoid(self.Uf(topics_embed.reshape(topics_embed.shape[0], -1).float()))

        queries = embeddings.permute([1, 0, 2])[-1].unsqueeze(0)

        inputs = queries.permute([1, 0, 2]).unbind(0)
        output_states = []
        attn_weight = []
        for i in range(len(inputs)):
            output, hidden, score, coverage_vector = self.decoder(input=inputs[i].unsqueeze(0),
                                                                  output=output,
                                                                  hidden=hidden,
                                                                  phi=phi,
                                                                  topics=topics_embed,
                                                                  coverage_vector=coverage_vector)  # [seq_len x batch x embed_size]
            output_states += [output]
            attn_weight += [score]

        output_states = torch.stack(output_states)
        attn_weight = torch.stack(attn_weight)

        outputs = self.adaptiveSoftmax.log_prob(output_states.reshape(-1, output_states.shape[-1]))
        return outputs, output_states, hidden, attn_weight, coverage_vector

    @torch.jit.export
    def cell_h(self, x):
        pass

    @torch.jit.export
    def cell_f(self, x, h, c):
        pass

    @torch.jit.export
    def predict_rnn(self, topics, num_chars, idx_to_word, word_to_idx):
        output_idx = [1]
        topics = [word_to_idx[x] for x in topics]
        topics = torch.tensor(topics)
        topics = topics.reshape((1, topics.shape[0]))
        #     hidden = torch.zeros(num_layers, 1, hidden_dim)
        #     hidden = (torch.zeros(num_layers, 1, hidden_dim).to(device), torch.zeros(num_layers, 1, hidden_dim).to(device))
        hidden = self.init_hidden(batch_size=1)
        adaptiveSoftmax = nn.AdaptiveLogSoftmaxWithLoss(1000, self.vocab_size, cutoffs=[round(self.vocab_size / 20),
                                                                                        4 * round(
                                                                                            self.vocab_size / 20)])
        if config.use_gpu:
            #         hidden = hidden.cuda()
            adaptiveSoftmax.to(self.device)
            topics = topics.to(self.device)
        coverage_vector = self.init_coverage_vector(topics.shape[0], topics.shape[1])
        attentions = torch.zeros(num_chars, topics.shape[1])
        for t in range(num_chars):
            X = torch.tensor(output_idx[-1]).reshape((1, 1))
            #         X = torch.tensor(output).reshape((1, len(output)))
            if config.use_gpu:
                X = X.to(self.device)
            if t == 0:
                output = torch.zeros(1, self.hidden_dim).to(self.device)
            else:
                output = output.squeeze(0)
            pred, output, hidden, attn_weight, coverage_vector = self.inference(inputs=X, topics=topics, output=output,
                                                                                hidden=hidden,
                                                                                coverage_vector=coverage_vector,
                                                                                seq_length=torch.tensor(50).reshape(1,
                                                                                                                    1).to(
                                                                                    self.device))
            #         print(coverage_vector)
            pred = pred.argmax(dim=1)  # greedy strategy
            attentions[t] = attn_weight[0].data
            #         pred = adaptive_softmax.predict(pred)
            if pred[-1] == 2:
                #         if pred.argmax(dim=1)[-1] == 2:
                break
            else:
                output_idx.append(int(pred[-1]))
        #             output.append(int(pred.argmax(dim=1)[-1]))
        return (''.join([idx_to_word[i] for i in output_idx[1:]]), [idx_to_word[i] for i in output_idx[1:]],
                attentions[:t + 1].t(), output_idx[1:])

    """Beamsearch strategy"""
    @torch.jit.export
    def beam_search(self, topics, num_chars, idx_to_word, word_to_idx, is_sample=False):
        output_idx = [1]
        topics = [word_to_idx[x] for x in topics]
        topics = torch.tensor(topics)
        topics = topics.reshape((1, topics.shape[0]))
        #     hidden = torch.zeros(num_layers, 1, hidden_dim)
        #     hidden = (torch.zeros(num_layers, 1, hidden_dim).to(device), torch.zeros(num_layers, 1, hidden_dim).to(device))
        hidden = self.init_hidden(batch_size=1)
        adaptiveSoftmax = nn.AdaptiveLogSoftmaxWithLoss(1000, self.vocab_size, cutoffs=[round(self.vocab_size / 20),
                                                                                        4 * round(
                                                                                            self.vocab_size / 20)])
        if config.use_gpu:
            #         hidden = hidden.cuda()
            adaptiveSoftmax.to(self.device)
            topics = topics.to(self.device)
            seq_length = torch.tensor(50).reshape(1, 1).to(self.device)
        """1"""
        coverage_vector = self.init_coverage_vector(topics.shape[0], topics.shape[1])
        attentions = torch.zeros(num_chars, topics.shape[1])
        X = torch.tensor(output_idx[-1]).reshape((1, 1)).to(self.device)
        output = torch.zeros(1, self.hidden_dim).to(self.device)
        log_prob, output, hidden, attn_weight, coverage_vector = self.inference(inputs=X,
                                                                                topics=topics,
                                                                                output=output,
                                                                                hidden=hidden,
                                                                                coverage_vector=coverage_vector,
                                                                                seq_length=seq_length)
        log_prob = log_prob.cpu().detach().reshape(-1).numpy()
        #     print(log_prob[10])
        """2"""
        if is_sample:
            top_indices = np.random.choice(self.vocab_size, config.beam_size, replace=False, p=np.exp(log_prob))
        else:
            top_indices = np.argsort(-log_prob)
        """3"""
        beams = [
            (0.0, [idx_to_word[1]], idx_to_word[1], torch.zeros(1, topics.shape[1]), torch.ones(1, topics.shape[1]))]
        b = beams[0]
        beam_candidates = []
        #     print(attn_weight[0].cpu().data, coverage_vector)
        #     assert False
        for i in range(config.beam_size):
            word_idx = top_indices[i]
            beam_candidates.append((b[0] + log_prob[word_idx], b[1] + [idx_to_word[word_idx]], word_idx,
                                    torch.cat((b[3], attn_weight[0].cpu().data), 0),
                                    torch.cat((b[4], coverage_vector.cpu().data), 0), hidden, output.squeeze(0),
                                    coverage_vector))
        """4"""
        beam_candidates.sort(key=lambda x: x[0], reverse=True)  # decreasing order
        beams = beam_candidates[:config.beam_size]  # truncate to get new beams

        for xy in range(num_chars - 1):
            beam_candidates = []
            for b in beams:
                """5"""
                X = torch.tensor(b[2]).reshape((1, 1)).to(self.device)
                """6"""
                log_prob, output, hidden, attn_weight, coverage_vector = self.inference(inputs=X,
                                                                                        topics=topics,
                                                                                        output=b[6],
                                                                                        hidden=b[5],
                                                                                        coverage_vector=b[7],
                                                                                        seq_length=seq_length)
                log_prob = log_prob.cpu().detach().reshape(-1).numpy()
                """8"""
                if is_sample:
                    top_indices = np.random.choice(self.vocab_size, config.beam_size, replace=False, p=np.exp(log_prob))
                else:
                    top_indices = np.argsort(-log_prob)
                """9"""
                for i in range(config.beam_size):
                    word_idx = top_indices[i]
                    beam_candidates.append((b[0] + log_prob[word_idx], b[1] + [idx_to_word[word_idx]], word_idx,
                                            torch.cat((b[3], attn_weight[0].cpu().data), 0),
                                            torch.cat((b[4], coverage_vector.cpu().data), 0), hidden, output.squeeze(0),
                                            coverage_vector))
            """10"""
            beam_candidates.sort(key=lambda x: x[0], reverse=True)  # decreasing order
            beams = beam_candidates[:config.beam_size]  # truncate to get new beams

        """11"""
        if '<EOS>' in beams[0][1]:
            first_eos = beams[0][1].index('<EOS>')
        else:
            first_eos = num_chars - 1
        return (
            ''.join(beams[0][1][:first_eos]), beams[0][1][:first_eos], beams[0][3][:first_eos].t(),
            beams[0][4][:first_eos])

    @torch.jit.export
    def init_hidden(self, batch_size):
        #         hidden = torch.zeros(num_layers, batch_size, hidden_dim)
        #         hidden = LSTMState(torch.zeros(batch_size, hidden_dim).to(device), torch.zeros(batch_size, hidden_dim).to(device))
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))
        return hidden

    @torch.jit.export
    def init_coverage_vector(self, batch_size, num_keywords):
        #         self.coverage_vector = torch.ones([batch_size, num_keywords]).to(device)
        return torch.ones([batch_size, num_keywords]).to(self.device)


"""Greedy decode strategy"""

@torch.jit.export
def pad_topic(topics, w2i):
    topics = [w2i[x] for x in topics]
    topics = torch.tensor(topics)
    print(topics)
    max_num = 5
    size = 1
    ans = np.zeros((size, max_num), dtype=int)
    for i in range(size):
        true_len = min(len(topics), max_num)
        for j in range(true_len):
            print(topics[i])
            ans[i][j] = topics[i][j]
    return ans
