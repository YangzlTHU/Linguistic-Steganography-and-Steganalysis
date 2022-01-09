import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


class Attention(nn.Module):
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size, embed_size):
        super(Attention, self).__init__()

        self.Ua = nn.Linear(embed_size, hidden_size, bias=False)
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.va = nn.Linear(hidden_size, 1, bias=True)
        # to store attention scores
        self.alphas = None

    def forward(self, query, topics, coverage_vector):
        scores = []
        C_t = coverage_vector.clone()
        for i in range(topics.shape[1]):
            proj_key = self.Ua(topics[:, i, :])
            query = self.Wa(query)
            scores += [self.va(torch.tanh(query + proj_key)) * C_t[:, i:i + 1]]

        # stack scores
        scores = torch.stack(scores, dim=1)
        scores = scores.squeeze(2)
        #         print(scores.shape)
        # turn scores to probabilities
        alphas = F.softmax(scores, dim=1)
        self.alphas = alphas

        # mt vector is the weighted sum of the topics
        mt = torch.bmm(alphas.unsqueeze(1), topics)
        mt = mt.squeeze(1)

        # mt shape: [batch x embed], alphas shape: [batch x num_keywords]
        return mt, alphas


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, embed_size, num_layers, dropout=0.5):
        super(AttentionDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.dropout = dropout

        # topic attention
        self.attention = Attention(hidden_size, embed_size)

        # lstm
        self.rnn = nn.LSTM(input_size=embed_size * 2,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=dropout)

    def forward(self, input, output, hidden, phi, topics, coverage_vector):
        # 1. calculate attention weight and mt
        mt, score = self.attention(output.squeeze(0), topics, coverage_vector)
        mt = mt.unsqueeze(1).permute(1, 0, 2)

        # 2. update coverge vector [batch x num_keywords]
        coverage_vector = coverage_vector - score / phi

        # 3. concat input and Tt, and feed into rnn
        output, hidden = self.rnn(torch.cat([input, mt], dim=2), hidden)

        return output, hidden, score, coverage_vector