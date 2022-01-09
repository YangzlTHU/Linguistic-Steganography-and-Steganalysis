import os
import sys
import json
import torch.nn as nn

class Config:
    def __init__(self):
        self.embedding_dim = 100
        self.hidden_dim = 512
        self.lr = 1e-3 * 0.5
        self.momentum = 0.01
        self.num_epoch = 100
        self.clip_value = 0.1
        self.use_gpu = True
        self.num_layers = 2
        self.bidirectional = False
        self.batch_size = 32
        self.num_keywords = 3
        self.verbose = 1
        self.check_point = 5
        self.beam_size = 2
        self.is_sample = True

config = Config()