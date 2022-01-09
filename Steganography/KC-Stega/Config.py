import os
import sys
import json

class Config:
    def __init__(self, test_file):
        self.result_dir = "./result"
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.summary_dir = "./summary/"
        os.makedirs(self.summary_dir, exist_ok=True)
        
        self.checkpoint_dir = self.result_dir + "/checkpoint"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_dir = "/best"
        self.tmp_model_dir = "/tmp"

        self.attribute_dir = "./data/attribute_json_modified"
        self.id_list = self.attribute_dir + "/val_id0.5.json"

        self.bit_file = "./data/bit_stream/bitstream_2000.json"

        self.data_dir = "./data/processed"
        self.train_file = self.data_dir + "/train.json"
        self.dev_file = self.data_dir + "/dev.jsonl"
        #self.test_file = self.data_dir + "/test.jsonl"
        self.test_file = self.data_dir + "/topics/{}.jsonl".format(test_file)
        self.make_test_file = self.data_dir + "/topics/hand_2topic_th0.5_5000data_921.jsonl"
        #self.make_test_file = self.data_dir + "/topics/gen3time_2topic_921.jsonl"
        self.wordvec_file = self.data_dir + "/wordvec.txt"
        self.vocab_file = self.data_dir + "/vocab.pkl"
        self.skip_cnt = 1
        self.brand_set_file = self.data_dir + "/brand_set.pkl"
        self.category_file = self.data_dir + "/category.pkl"
        self.feat_key_file = self.data_dir + "/feat_key.pkl"
        self.feat_val_file = self.data_dir + "/feat_val.pkl"
        self.cateFK2val_file = self.data_dir + "/cateFK2val.pkl"

        # dataset
        self.word_dim = 300
        self.bucket_width = 10
        self.num_buckets = 5
        self.shuffle_buffer_size = 10000

        # overall
        self.epoch = 100
        self.num_training_step = 900000
        self.train_batch_size = 32
        self.test_batch_size = 1 # 变大无法进行判断
        self.steps_per_stat = 10

        # Steganography
        self.bit_per_word = 2
        self.AC_precision = 26
        self.truncated_vocab_size=5000

    def check_ckpt(self, model_name):
        if not os.path.exists(self.checkpoint_dir + "/" + model_name + "/"):
            os.mkdir(self.checkpoint_dir + "/" + model_name + "/")
        if not os.path.exists(self.checkpoint_dir + "/" + model_name + self.best_model_dir):
            os.mkdir(self.checkpoint_dir + "/" + model_name + self.best_model_dir)
        if not os.path.exists(self.checkpoint_dir + "/" + model_name + self.tmp_model_dir):
            os.mkdir(self.checkpoint_dir + "/" + model_name + self.tmp_model_dir)

config = Config('test0')
