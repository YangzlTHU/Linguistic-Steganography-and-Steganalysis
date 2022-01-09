import tensorflow as tf
import numpy as np
#np.set_printoptions(threshold=np.inf)
import os
import sys
import collections
from Models import model_utils
from Config import config
from Models import PHVM_SampleEmbeddingHelper
from Models.RS import PHVM_RSEmbeddingHelper
from Models.RS.basic_decoder import BasicDecoder as RS_BasicDecoder
from Models.RS.decoder import dynamic_decode as RS_dynamic_decode
from Models.AC import AC_SampleEmbeddingHelper
from Models.AC.basic_decoder import BasicDecoder as AC_BasicDecoder
from Models.AC.decoder import dynamic_decode as AC_dynamic_decode


#使用namedtuple实现一种数据结构：PHVMBatchInput.key_input/.val_input/...
class PHVMBatchInput(collections.namedtuple("PHVMBatchInput",
                                          ("key_input", "val_input", "val_word", "input_lens",
                                           "target_input", "target_output", "output_lens",
                                           "group", "group_lens", "group_cnt",
                                           "target_type", "target_type_lens",
                                           "text", "slens",
                                           "category"))):
    pass

class PHVMConfig:
    def __init__(self):
        # rnn--Bidirectional GRU
        self.PHVM_rnn_direction = 'bi'
        self.PHVM_rnn_type = 'gru'

        # embedding
        self.share_vocab = False
        self.PHVM_word_dim = 300
        self.PHVM_key_dim = 30
        self.PHVM_val_dim = 100
        self.PHVM_cate_dim = 10

        # group
        self.PHVM_group_selection_threshold = 0.5#选择某个d_i是否在g_t中的概率下限
        self.PHVM_stop_threshold = 0.5#停止g_i的编码信号
        self.PHVM_max_group_cnt = 30
        self.PHVM_max_sent_cnt = 10

        # type
        self.PHVM_use_type_info = False
        self.PHVM_type_dim = 30

        # encoder
        self.PHVM_encoder_dim = 100
        self.PHVM_encoder_num_layer = 1

        # group_decoder
        self.PHVM_group_decoder_dim = 100
        self.PHVM_group_decoder_num_layer = 1

        # group encoder
        self.PHVM_group_encoder_dim = 100
        self.PHVM_group_encoder_num_layer = 1

        # latent_decoder
        self.PHVM_latent_decoder_dim = 300
        self.PHVM_latent_decoder_num_layer = 1

        # sent_top_encoder
        self.PHVM_sent_top_encoder_dim = 300
        self.PHVM_sent_top_encoder_num_layer = 1

        # text post encoder
        self.PHVM_text_post_encoder_dim = 300
        self.PHVM_text_post_encoder_num_layer = 1

        # sent_post_encoder
        self.PHVM_sent_post_encoder_dim = 300
        self.PHVM_sent_post_encoder_num_layer = 1

        # bow--average pooling
        self.PHVM_bow_hidden_dim = 200

        # decoder
        self.PHVM_decoder_dim = 300
        self.PHVM_decoder_num_layer = 2

        # latent
        self.PHVM_plan_latent_dim = 200
        self.PHVM_sent_latent_dim = 200

        # training
        self.PHVM_learning_rate = 0.001
        self.PHVM_num_training_step = 100000
        self.PHVM_sent_full_KL_step = 20000
        self.PHVM_plan_full_KL_step = 40000
        self.PHVM_dropout = 0

        # inference
        self.PHVM_beam_width = 1 # 原来是10
        self.PHVM_maximum_iterations = 50

        # hide information
        self.PHVM_bpw = config.bit_per_word
        self.AC_precision = config.AC_precision
        self.truncated_vocab_size = config.truncated_vocab_size

class PHVM:
    def __init__(self, key_vocab_size, val_vocab_size, tgt_vocab_size, cate_vocab_size, hide_strategy, STRATEGY,
                 key_wordvec=None, val_wordvec=None, tgt_wordvec=None,
                 type_vocab_size=None, start_token=0, end_token=1, config=PHVMConfig()):
        self.config = config
        self.key_vocab_size = key_vocab_size
        self.val_vocab_size = val_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.cate_vocab_size = cate_vocab_size
        self.hide_strategy = hide_strategy
        self.STRATAGY = STRATEGY
        self.type_vocab_size = type_vocab_size
        self.start_token = start_token
        self.end_token = end_token
        self.early_stopping = 15
        self.truncated_vocab_size = config.truncated_vocab_size
        self.indices = model_utils.get_bit_group(config.PHVM_bpw, self.truncated_vocab_size)
        self.AC_indices = model_utils.get_bit_group(config.AC_precision, self.truncated_vocab_size)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = self.get_input_tuple()
            self.bitfile = tf.placeholder(shape=[None], dtype=tf.int32)
            self.no_hide = tf.placeholder(shape=[None], dtype=tf.int32)
            self.build_graph(key_wordvec, val_wordvec, tgt_wordvec)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.init = tf.global_variables_initializer()
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)
            self.best_saver = tf.train.Saver()
            self.tmp_saver = tf.train.Saver()

    def get_input_tuple(self):
        return PHVMBatchInput(#placeholder占位符当形参
            key_input=tf.placeholder(shape=[None, None], dtype=tf.int32),
            val_input=tf.placeholder(shape=[None, None], dtype=tf.int32),
            val_word=tf.placeholder(shape=[None], dtype=tf.int32),
            input_lens=tf.placeholder(shape=[None], dtype=tf.int32),

            target_input=tf.placeholder(shape=[None, None, None], dtype=tf.int32),
            target_output=tf.placeholder(shape=[None, None, None], dtype=tf.int32),
            output_lens=tf.placeholder(shape=[None, None], dtype=tf.int32),

            group=tf.placeholder(shape=[None, None, None], dtype=tf.int32),
            group_lens=tf.placeholder(shape=[None, None], dtype=tf.int32),
            group_cnt=tf.placeholder(shape=[None], dtype=tf.int32),

            target_type=tf.placeholder(shape=[None, None, None], dtype=tf.int32),
            target_type_lens=tf.placeholder(shape=[None, None], dtype=tf.int32),

            text=tf.placeholder(shape=[None, None], dtype=tf.int32),
            slens=tf.placeholder(shape=[None], dtype=tf.int32),

            category=tf.placeholder(shape=[None], dtype=tf.int32)
        )

    def get_learning_rate(self):
        self.learning_rate = tf.constant(self.config.PHVM_learning_rate, dtype=tf.float32)
        start_decay_step = self.config.PHVM_num_training_step // 2
        decay_times = 5
        decay_factor = 0.5
        decay_steps = (self.config.PHVM_num_training_step - start_decay_step) // decay_times
        return tf.cond(#条件是：训练次数<开始衰减次数
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                tf.minimum(self.global_step - start_decay_step, 3 * decay_steps),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")

    def sample_gaussian(self, shape, mu, logvar):#变分后的高斯分布下采样
        x = tf.random_normal(shape, dtype=tf.float32)
        return tf.cond(tf.equal(len(shape), 2),
                       lambda: mu + tf.exp(logvar / 2) * x,
                       lambda: tf.expand_dims(mu, 1) + tf.exp(tf.expand_dims(logvar / 2, 1)) * x)

    def make_embedding(self, key_wordvec, val_wordvec, tgt_wordvec):
        if tgt_wordvec is None:
            self.word_embedding = tf.get_variable("word_embedding",
                                                  shape=[self.tgt_vocab_size, self.config.PHVM_word_dim],
                                                  dtype=tf.float32)
        else:
            self.word_embedding = tf.get_variable("word_embedding", dtype=tf.float32,
                                                  initializer=tf.constant(tgt_wordvec, dtype=tf.float32))
        #val embedding
        if self.config.share_vocab:
            self.val_embedding = self.word_embedding
        else:
            if val_wordvec is None:
                self.val_embedding = tf.get_variable("val_embedding",
                                                     shape=[self.val_vocab_size, self.config.PHVM_val_dim],
                                                     dtype=tf.float32)
            else:
                self.val_embedding = tf.get_variable("val_embedding", dtype=tf.float32,
                                                     initializer=tf.constant(val_wordvec, dtype=tf.float32))
        #key embedding
        if key_wordvec is None:
            self.key_embedding = tf.get_variable("key_embedding",
                                                 shape=[self.key_vocab_size, self.config.PHVM_key_dim],
                                                 dtype=tf.float32)
        else:
            self.key_embedding = tf.get_variable("key_embedding", dtype=tf.float32,
                                                 initializer=tf.constant(key_wordvec, dtype=tf.float32))

        self.cate_embedding = tf.get_variable("cate_embedding",
                                              shape=[self.cate_vocab_size, self.config.PHVM_cate_dim],
                                              dtype=tf.float32)

        if self.config.PHVM_use_type_info:
            self.type_embedding = tf.get_variable("type_embedding",
                                                  shape=[self.type_vocab_size, self.config.PHVM_type_dim],
                                                  dtype=tf.float32)
    #先验和后验分布的KL散度计算
    def KL_divergence(self, prior_mu, prior_logvar, post_mu, post_logvar, reduce_mean=True):
        divergence = 0.5 * tf.reduce_sum(tf.exp(post_logvar - prior_logvar)
                                         + tf.pow(post_mu - prior_mu, 2) / tf.exp(prior_logvar)
                                         - 1 - (post_logvar - prior_logvar), axis=1)
        if reduce_mean:
            return tf.reduce_sum(divergence)
        else:
            return divergence

    def gather_group(self, src, group_idx, group_lens, group_cnt, group_encoder):
        shape = tf.shape(group_idx)
        batch_size = shape[0]
        fidx = tf.expand_dims(tf.expand_dims(tf.range(batch_size), 1), 2)
        fidx = tf.tile(fidx, [1, shape[1], shape[2]])
        fidx = tf.expand_dims(fidx, 3)
        sidx = tf.expand_dims(group_idx, 3)
        gidx = tf.concat((fidx, sidx), 3)
        group_bow = tf.gather_nd(src, gidx)
        group_mask = tf.sequence_mask(group_lens, shape[2], dtype=tf.float32)
        expanded_group_mask = tf.expand_dims(group_mask, 3)
        group_sum_bow = tf.reduce_sum(group_bow * expanded_group_mask, 2)
        safe_group_lens = group_lens + tf.cast(tf.equal(group_lens, 0), dtype=tf.int32)
        group_mean_bow = group_sum_bow / tf.to_float(tf.expand_dims(safe_group_lens, 2))

        group_encoder_output, group_encoder_state = tf.nn.dynamic_rnn(group_encoder,
                                                                      group_mean_bow,
                                                                      group_cnt,
                                                                      dtype=tf.float32)

        if self.config.PHVM_rnn_type == 'lstm':
            group_embed = group_encoder_state.h#多了一个cell state
        else:
            group_embed = group_encoder_state
        return gidx, group_bow, group_mean_bow, group_embed

    def build_graph(self, key_wordvec, val_wordvec, tgt_wordvec):
        self.global_step = tf.get_variable("global_step", dtype=tf.int32, initializer=tf.constant(0), trainable=False)
        self.keep_prob = tf.placeholder(shape=[], dtype=tf.float32)
        self.train_flag = tf.placeholder(shape=[], dtype=tf.bool)
        self.batch_size = tf.size(self.input.input_lens)

        with tf.variable_scope("embedding"):
            self.make_embedding(key_wordvec, val_wordvec, tgt_wordvec)

            #这里两个就是二元组的[key, val]嵌入过程
            key_embed = tf.nn.embedding_lookup(self.key_embedding, self.input.key_input)
            val_embed = tf.nn.embedding_lookup(self.val_embedding, self.input.val_input)
            src = tf.concat((key_embed, val_embed), 2)
            cate_embed = tf.nn.embedding_lookup(self.cate_embedding, self.input.category)
            if self.config.PHVM_use_type_info:
                type_embed = tf.nn.embedding_lookup(self.type_embedding, self.input.target_type)
                type_mask = tf.sequence_mask(self.input.target_type_lens,
                                             tf.shape(self.input.target_type)[2],
                                             dtype=tf.float32)
                expanded_type_mask = tf.expand_dims(type_mask, 3)
                type_embed = tf.reduce_sum(type_embed * expanded_type_mask, 2)
                safe_target_type_lens = self.input.target_type_lens + \
                                        tf.cast(tf.equal(self.input.target_type_lens, 0), dtype=tf.int32)
                type_embed = type_embed / tf.cast(tf.expand_dims(safe_target_type_lens, 2), dtype=tf.float32)

            text = tf.nn.embedding_lookup(self.word_embedding, self.input.text)

        with tf.variable_scope("input_encode"):
            if self.config.PHVM_rnn_direction == 'uni':
                src_cell = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                    self.config.PHVM_encoder_num_layer,
                                                    self.config.PHVM_encoder_dim,
                                                    self.keep_prob,
                                                    "src_encoder")
                src_encoder_output, src_encoder_state = tf.nn.dynamic_rnn(src_cell,
                                                                          src,
                                                                          self.input.input_lens,
                                                                          dtype=tf.float32)
                if self.config.PHVM_rnn_type == 'lstm':
                    src_embed = src_encoder_state.h
                else:
                    src_embed = src_encoder_state
            else:#当encoder为双向rnn时
                #前向forward
                src_fw_cell = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                       self.config.PHVM_encoder_num_layer,
                                                       self.config.PHVM_encoder_dim,
                                                       self.keep_prob,
                                                       "src_fw_encoder")
                #后向backward
                src_bw_cell = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                       self.config.PHVM_encoder_num_layer,
                                                       self.config.PHVM_encoder_dim,
                                                       self.keep_prob,
                                                       "src_bw_encoder")
                src_encoder_output, src_encoder_state = tf.nn.bidirectional_dynamic_rnn(src_fw_cell,
                                                                                        src_bw_cell,
                                                                                        src,
                                                                                        self.input.input_lens,
                                                                                        dtype=tf.float32)
                src_encoder_output = tf.concat(src_encoder_output, 2)
                if self.config.PHVM_rnn_type == 'lstm':
                    src_embed = tf.concat((src_encoder_state[0].h, src_encoder_state[1].h), 1)#两层rnn的最后一个hidden state拼接
                else:
                    src_embed = tf.concat((src_encoder_state[0], src_encoder_state[1]), 1)

        with tf.variable_scope("text_encode"):
            #textencoder默认为双向rnn
            tgt_fw_cell = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                   self.config.PHVM_text_post_encoder_num_layer,
                                                   self.config.PHVM_text_post_encoder_dim,
                                                   self.keep_prob,
                                                   "text_fw_encoder")
            tgt_bw_cell = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                   self.config.PHVM_text_post_encoder_num_layer,
                                                   self.config.PHVM_text_post_encoder_dim,
                                                   self.keep_prob,
                                                   "text_bw_encoder")
            tgt_encoder_output, tgt_encoder_state = tf.nn.bidirectional_dynamic_rnn(tgt_fw_cell,
                                                                                    tgt_bw_cell,
                                                                                    text,
                                                                                    self.input.slens,
                                                                                    dtype=tf.float32)
            if self.config.PHVM_rnn_type == 'lstm':
                #lstm的state由tuple(c, h)组成（在vanilla RNN中直接state=hidden），其中state.h代表某个时间步的hidden state进行embedding
                tgt_embed = tf.concat((tgt_encoder_state[0].h, tgt_encoder_state[1].h), 1)
            else:
                #GRU中state直接就等于h，所以无需指定state.h
                tgt_embed = tf.concat((tgt_encoder_state[0], tgt_encoder_state[1]), 1)

        with tf.variable_scope("top_level"):
            #分别用MLP拟合先验和后验分布的参数
            with tf.variable_scope("prior_network"):
                prior_input = tf.concat((cate_embed, src_embed), 1)#输入的是二元组的embedding
                prior_fc = tf.layers.dense(prior_input, self.config.PHVM_plan_latent_dim * 2, activation=tf.tanh)
                prior_fc_nd = tf.layers.dense(prior_fc, self.config.PHVM_plan_latent_dim * 2)#这里乘了2！！相当于mu和logvar放在同一个网络里面在训练啦！！！
                prior_mu, prior_logvar = tf.split(prior_fc_nd, 2, 1)# 把prior_fc_nd这个张量 平均 分为2份了分别作为 均值 和 方差（看上面的解释！）
                prior_z_plan = self.sample_gaussian((self.batch_size, self.config.PHVM_plan_latent_dim),
                                                    prior_mu,
                                                    prior_logvar)

            with tf.variable_scope("posterior_network"):
                post_input = tf.concat((cate_embed, src_embed, tgt_embed), 1)
                post_fc = tf.layers.dense(post_input, self.config.PHVM_plan_latent_dim * 2)# 全连接拟合分布方差与均值
                post_mu, post_logvar = tf.split(post_fc, 2, 1)
                post_z_plan = self.sample_gaussian((self.batch_size, self.config.PHVM_plan_latent_dim),
                                                   post_mu,
                                                   post_logvar)

            self.plan_KL_divergence = self.KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar)
            #指定KL散度的weight
            plan_KL_weight = tf.minimum(1.0, tf.to_float(self.global_step) / self.config.PHVM_plan_full_KL_step)

            dec_input = tf.cond(self.train_flag,
                                lambda: tf.concat((cate_embed, src_embed, post_z_plan), 1),
                                lambda: tf.concat((cate_embed, src_embed, prior_z_plan), 1))
            if self.config.PHVM_rnn_direction == 'uni':
                dec_input_dim = self.config.PHVM_cate_dim + self.config.PHVM_encoder_dim + \
                                self.config.PHVM_plan_latent_dim
            else:
                dec_input_dim = self.config.PHVM_cate_dim + 2 * self.config.PHVM_encoder_dim + \
                                self.config.PHVM_plan_latent_dim
            dec_input = tf.reshape(dec_input, [self.batch_size, dec_input_dim])

        with tf.variable_scope("sentence_level"):

            with tf.variable_scope("group_sent"):
                group_decoder = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                         self.config.PHVM_group_decoder_num_layer,
                                                         self.config.PHVM_group_decoder_dim,
                                                         self.keep_prob,
                                                         'group_decoder')

                group_encoder = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                         self.config.PHVM_group_encoder_num_layer,
                                                         self.config.PHVM_group_encoder_dim,
                                                         self.keep_prob,
                                                         "group_encoder")

                latent_decoder = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                          self.config.PHVM_latent_decoder_num_layer,
                                                          self.config.PHVM_latent_decoder_dim,
                                                          self.keep_prob,
                                                          "latent_decoder")

                decoder = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                   self.config.PHVM_decoder_num_layer,
                                                   self.config.PHVM_decoder_dim,
                                                   self.keep_prob,
                                                   "decoder")

            with tf.variable_scope("sent_post_encoder"):
                #双向RNN
                sent_fw_cell = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                        self.config.PHVM_sent_post_encoder_num_layer,
                                                        self.config.PHVM_sent_post_encoder_dim,
                                                        self.keep_prob,
                                                        "sent_fw_cell")
                sent_bw_cell = model_utils.get_rnn_cell(self.config.PHVM_rnn_type,
                                                        self.config.PHVM_sent_post_encoder_num_layer,
                                                        self.config.PHVM_sent_post_encoder_dim,
                                                        self.keep_prob,
                                                        "sent_bw_cell")

            with tf.variable_scope("parameters"):
                with tf.variable_scope("init_state"):
                    group_init_state_fc = tf.layers.Dense(self.config.PHVM_group_decoder_dim)
                    if self.config.PHVM_rnn_direction == 'uni':
                        init_gbow = tf.get_variable("start_of_group", dtype=tf.float32,
                                                    shape=(1, self.config.PHVM_encoder_dim))
                    else:
                        init_gbow = tf.get_variable("start_of_group", dtype=tf.float32,
                                                   shape=(1, 2 * self.config.PHVM_encoder_dim))
                    plan_init_state_fc = tf.layers.Dense(self.config.PHVM_latent_decoder_dim)

                prior_fc_layer = tf.layers.Dense(self.config.PHVM_sent_latent_dim * 2)
                post_fc_layer = tf.layers.Dense(self.config.PHVM_sent_latent_dim * 2)

                group_fc_1 = tf.layers.Dense(self.config.PHVM_encoder_dim)
                group_fc_2 = tf.layers.Dense(1)

                type_fc_1 = tf.layers.Dense(self.config.PHVM_type_dim)
                type_fc_2 = tf.layers.Dense(self.type_vocab_size)

                bow_fc_1 = tf.layers.Dense(self.config.PHVM_bow_hidden_dim)
                bow_fc_2 = tf.layers.Dense(self.tgt_vocab_size)

                projection = tf.layers.Dense(self.tgt_vocab_size)

                stop_clf = tf.layers.Dense(1)

            with tf.name_scope("train"):
                with tf.name_scope("group_encode"):
                    gidx, group_bow, group_mean_bow, group_embed = self.gather_group(src_encoder_output,
                                                                                     self.input.group,
                                                                                     self.input.group_lens,
                                                                                     self.input.group_cnt,
                                                                                     group_encoder)

                def train_cond(i, group_state, gbow, plan_state, sent_state, sent_z,
                               stop_sign, sent_rec_loss, group_rec_loss, KL_loss, type_loss, bow_loss):
                    return i < tf.shape(self.input.output_lens)[1]

                def train_body(i, group_state, gbow, plan_state, sent_state, sent_z,
                               stop_sign, sent_rec_loss, group_rec_loss, KL_loss, type_loss, bow_loss):
                    sent_input = tf.nn.embedding_lookup(self.word_embedding, self.input.target_input[:, i, :])
                    sent_output = self.input.target_output[:, i, :]
                    sent_lens = self.input.output_lens[:, i]
                    sent_mask = tf.sequence_mask(sent_lens, tf.shape(sent_output)[1], dtype=tf.float32)
                    loss_mask = 1 - tf.to_float(tf.equal(sent_lens, 0))
                    effective_cnt = tf.to_float(tf.reduce_sum(1 - tf.to_int32(tf.equal(sent_lens, 0))))

                    with tf.variable_scope("sent_encoder"):
                        cali_sent_lens = sent_lens - 1 + tf.cast(tf.equal(sent_lens, 0), dtype=tf.int32)
                        sent_encoder_output, sent_encoder_state = tf.nn.bidirectional_dynamic_rnn(sent_fw_cell,
                                                                                                  sent_bw_cell,
                                                                                                  sent_input[:, 1:, :],
                                                                                                  cali_sent_lens,
                                                                                                  dtype=tf.float32)
                        if self.config.PHVM_rnn_type == 'lstm':
                            sent_embed = tf.concat((sent_encoder_state[0].h, sent_encoder_state[1].h), 1)
                        else:
                            sent_embed = tf.concat((sent_encoder_state[0], sent_encoder_state[1]), 1)

                    with tf.name_scope("group"):
                        sent_gid = gidx[:, i, :, :]
                        sent_group = group_bow[:, i, :, :]
                        sent_group_len = self.input.group_lens[:, i]
                        safe_sent_group_len = sent_group_len + tf.cast(tf.equal(sent_group_len, 0), dtype=tf.int32)
                        group_mask = tf.sequence_mask(sent_group_len, tf.shape(sent_group)[1], dtype=tf.float32)
                        expanded_group_mask = tf.expand_dims(group_mask, 2)

                        with tf.variable_scope("decode_group"):
                            gout, group_state = group_decoder(gbow, group_state)
                            tile_gout = tf.tile(tf.expand_dims(gout, 1), [1, tf.shape(src_encoder_output)[1], 1])
                            group_fc_input = tf.concat((src_encoder_output, tile_gout), 2)
                            group_logit = tf.squeeze(group_fc_2(tf.tanh(group_fc_1(group_fc_input))), 2)
                            group_label = tf.one_hot(sent_gid[:, :, 1], tf.shape(group_logit)[1], dtype=tf.float32)
                            group_label = tf.reduce_sum(group_label * expanded_group_mask, 1)
                            group_crossent = tf.nn.sigmoid_cross_entropy_with_logits(labels=group_label,
                                                                                     logits=group_logit)
                            src_mask = tf.sequence_mask(self.input.input_lens, tf.shape(group_logit)[1],
                                                        dtype=tf.float32)
                            group_crossent = loss_mask * tf.reduce_sum(group_crossent * src_mask, 1)
                            group_rec_loss += tf.reduce_sum(group_crossent) # / effective_cnt

                        gbow = group_mean_bow[:, i, :]

                        with tf.name_scope("stop_loss"):
                            stop_sign = stop_sign.write(i, tf.squeeze(stop_clf(gout), axis=1))

                    if self.config.PHVM_use_type_info:
                        sent_type_embed = type_embed[:, i, :]
                        sent_type = self.input.target_type[:, i, :]
                        sent_type_len = self.input.target_type_lens[:, i]
                        sent_type_mask = tf.sequence_mask(sent_type_len, tf.shape(sent_type)[1], dtype=tf.float32)

                    # latent_decoder_input
                    if self.config.PHVM_rnn_type == 'lstm':
                        plan_input = tf.concat((sent_state.h, sent_z), 1)
                    else:
                        plan_input = tf.concat((sent_state, sent_z), 1)
                    sent_cond_embed, plan_state = latent_decoder(plan_input, plan_state)

                    with tf.name_scope("sent_prior_network"):
                        sent_prior_input = tf.concat((sent_cond_embed, gbow), 1)
                        sent_prior_fc = prior_fc_layer(sent_prior_input)
                        sent_prior_mu, sent_prior_logvar = tf.split(sent_prior_fc, 2, axis=1)

                    with tf.name_scope("sent_posterior_network"):
                        if self.config.PHVM_use_type_info:
                            sent_post_input = tf.concat((sent_cond_embed, gbow, sent_embed, sent_type_embed), 1)
                        else:
                            sent_post_input = tf.concat((sent_cond_embed, gbow, sent_embed), 1)
                        sent_post_fc = post_fc_layer(sent_post_input)
                        sent_post_mu, sent_post_logvar = tf.split(sent_post_fc, 2, axis=1)
                        sent_z = self.sample_gaussian((self.batch_size, self.config.PHVM_sent_latent_dim),
                                                      sent_post_mu,
                                                      sent_post_logvar)#隐变量采样
                        sent_z = tf.reshape(sent_z, (self.batch_size, self.config.PHVM_sent_latent_dim))

                    sent_cond_z_embed = tf.concat((sent_cond_embed, sent_z), 1)

                    with tf.name_scope("KL_divergence"):
                        divergence = loss_mask * self.KL_divergence(sent_prior_mu, sent_prior_logvar,
                                                                    sent_post_mu, sent_post_logvar, False)
                        KL_loss += tf.reduce_sum(divergence) # / effective_cnt

                    with tf.name_scope("type_loss"):
                        if self.config.PHVM_use_type_info:
                            type_input = tf.concat((sent_cond_z_embed, gbow), 1)
                            type_logit = type_fc_2(tf.tanh(type_fc_1(type_input)))
                            type_logit = tf.tile(tf.expand_dims(type_logit, 1), [1, tf.shape(sent_type)[1], 1])
                            type_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sent_type,
                                                                                           logits=type_logit)
                            type_crossent = loss_mask * tf.reduce_sum(type_crossent * sent_type_mask, 1)
                            type_loss += tf.reduce_sum(type_crossent) # / effective_cnt
                        else:
                            type_loss = 0.0

                    with tf.name_scope("sent_deocde"):
                        with tf.variable_scope("sent_dec_state"):
                            if self.config.PHVM_use_type_info:
                                sent_dec_input = tf.concat((sent_cond_z_embed, gbow, sent_type_embed), 1)#直接将相应的特征进行拼接
                            else:
                                sent_dec_input = tf.concat((sent_cond_z_embed, gbow), 1)
                            sent_dec_state = []
                            for _ in range(self.config.PHVM_decoder_num_layer):
                                tmp = tf.layers.dense(sent_dec_input, self.config.PHVM_decoder_dim)
                                if self.config.PHVM_rnn_type == 'lstm':
                                    tmp = tf.nn.rnn_cell.LSTMStateTuple(c=tmp, h=tmp)
                                sent_dec_state.append(tmp)
                            if self.config.PHVM_decoder_num_layer > 1:
                                sent_dec_state = tuple(sent_dec_state)
                            else:
                                sent_dec_state = sent_dec_state[0]

                        with tf.variable_scope("attention"):
                            attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.config.PHVM_decoder_dim,
                                                                                    sent_group,
                                                                                    memory_sequence_length=safe_sent_group_len)
                        train_decoder = tf.contrib.seq2seq.AttentionWrapper(decoder, attention_mechanism,
                                                                    attention_layer_size=self.config.PHVM_decoder_dim)
                        train_encoder_state = train_decoder.zero_state(self.batch_size, dtype=tf.float32).clone(
                            cell_state=sent_dec_state)
                        helper = tf.contrib.seq2seq.TrainingHelper(sent_input, sent_lens, time_major=False)
                        basic_decoder = tf.contrib.seq2seq.BasicDecoder(train_decoder, helper, train_encoder_state,
                                                                        output_layer=projection)
                        with tf.variable_scope("dynamic_decoding"):
                            fout, fstate, flens = tf.contrib.seq2seq.dynamic_decode(basic_decoder, impute_finished=True)
                        sent_logit = fout.rnn_output
                        cali_sent_output = sent_output[:, :tf.shape(sent_logit)[1]]
                        sent_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=cali_sent_output,
                                                                                       logits=sent_logit)
                        cali_sent_mask = sent_mask[:, :tf.shape(sent_logit)[1]]
                        sent_crossent = loss_mask * tf.reduce_sum(sent_crossent * cali_sent_mask, axis=1)
                        sent_rec_loss += tf.reduce_sum(sent_crossent) # / effective_cnt

                        with tf.name_scope("bow_loss"):
                            bow_logit = bow_fc_2(tf.tanh(bow_fc_1(sent_dec_input)))
                            bow_logit = tf.tile(tf.expand_dims(bow_logit, 1), [1, tf.shape(sent_output)[1], 1])
                            bow_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sent_output,
                                                                                          logits=bow_logit)
                            bow_crossent = loss_mask * tf.reduce_sum(bow_crossent * sent_mask, axis=1)
                            bow_loss += tf.reduce_sum(bow_crossent) # / effective_cnt

                        with tf.variable_scope("sent_state_update"):
                            sent_state = fstate.cell_state[self.config.PHVM_decoder_num_layer - 1]
                            # if self.config.PHVM_rnn_type == 'lstm':
                            #     sent_top_input = fstate.cell_state[self.config.PHVM_decoder_num_layer - 1].h
                            # else:
                            #     sent_top_input = fstate.cell_state[self.config.PHVM_decoder_num_layer - 1]
                            # _, sent_state = sent_top_encoder(sent_top_input, sent_state)

                    return i + 1, group_state, gbow, plan_state, sent_state, sent_z, \
                           stop_sign, sent_rec_loss, group_rec_loss, KL_loss, type_loss, bow_loss

                group_state = group_init_state_fc(dec_input)
                gbow = tf.tile(init_gbow, [self.batch_size, 1])
                plan_state = plan_init_state_fc(tf.concat((dec_input, group_embed), 1))
                sent_state = decoder.zero_state(self.batch_size, dtype=tf.float32)[
                    self.config.PHVM_decoder_num_layer - 1]
                sent_z = tf.zeros(shape=(self.batch_size, self.config.PHVM_sent_latent_dim), dtype=tf.float32)
                stop_sign = tf.TensorArray(dtype=tf.float32, element_shape=(None,),
                                           size=tf.shape(self.input.group)[1])

                _, group_state, gbow, plan_state, sent_state, sent_z, \
                stop_sign, sent_rec_loss, group_rec_loss, KL_loss, type_loss, bow_loss = \
                    tf.while_loop(train_cond, train_body,
                                  loop_vars=(0, group_state, gbow, plan_state, sent_state, sent_z, stop_sign,
                                             0.0, 0.0, 0.0, 0.0, 0.0))

                with tf.name_scope("loss_computation"):
                    stop_logit = tf.transpose(stop_sign.stack(), [1, 0])
                    stop_label = tf.one_hot(self.input.group_cnt - 1, tf.shape(stop_logit)[1])
                    stop_crossent = tf.nn.sigmoid_cross_entropy_with_logits(logits=stop_logit, labels=stop_label)
                    stop_mask = tf.sequence_mask(self.input.group_cnt, tf.shape(stop_logit)[1], dtype=tf.float32)
                    self.stop_loss = tf.reduce_mean(tf.reduce_sum(stop_crossent * stop_mask, 1))

                    self.sent_rec_loss = sent_rec_loss
                    self.group_rec_loss = group_rec_loss
                    self.sent_KL_divergence = KL_loss
                    sent_KL_weight = tf.minimum(1.0, tf.to_float(self.global_step) / self.config.PHVM_sent_full_KL_step)
                    self.type_loss = type_loss
                    self.bow_loss = bow_loss / tf.to_float(self.batch_size)
                    anneal_sent_KL = sent_KL_weight * self.sent_KL_divergence
                    anneal_plan_KL = plan_KL_weight * self.plan_KL_divergence
                    self.elbo_loss = self.sent_rec_loss + self.group_rec_loss + self.type_loss + \
                                     self.sent_KL_divergence + self.plan_KL_divergence
                    self.elbo_loss /= tf.to_float(self.batch_size)
                    self.anneal_elbo_loss = self.sent_rec_loss + self.group_rec_loss + self.type_loss + \
                                            anneal_sent_KL + anneal_plan_KL
                    self.anneal_elbo_loss /= tf.to_float(self.batch_size)
                    self.train_loss = self.anneal_elbo_loss + self.stop_loss + self.bow_loss

                with tf.name_scope("update"):
                    params = tf.trainable_variables()
                    gradients = tf.gradients(self.train_loss, params)
                    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, 5)

                    self.learning_rate = self.get_learning_rate()
                    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    self.update = self.optimizer.apply_gradients(zip(clipped_gradients, params),
                                                                 global_step=self.global_step)

                with tf.name_scope("summary"):#到tensorboard
                    self.gradient_summary = [tf.summary.scalar("gradient_norm", gradient_norm),
                                             tf.summary.scalar("clipped gradient_norm", tf.global_norm(clipped_gradients))]
                    self.train_summary = tf.summary.merge([tf.summary.scalar("learning rate", self.learning_rate),
                                                           tf.summary.scalar("train_loss", self.train_loss),
                                                           tf.summary.scalar("elbo", self.elbo_loss),
                                                           tf.summary.scalar("sent_KL_divergence", self.sent_KL_divergence / tf.to_float(self.batch_size)),
                                                           tf.summary.scalar("anneal_sent_KL", anneal_sent_KL / tf.to_float(self.batch_size)),
                                                           tf.summary.scalar("anneal_plan_KL", anneal_plan_KL / tf.to_float(self.batch_size)),
                                                           tf.summary.scalar("plan_KL_divergence", self.plan_KL_divergence / tf.to_float(self.batch_size)),
                                                           tf.summary.scalar("sent_rec_loss", self.sent_rec_loss / tf.to_float(self.batch_size)),
                                                           tf.summary.scalar("group_rec_loss", self.group_rec_loss / tf.to_float(self.batch_size)),
                                                           tf.summary.scalar("type_loss", self.type_loss / tf.to_float(self.batch_size)),
                                                           tf.summary.scalar("stop_loss", self.stop_loss),
                                                           tf.summary.scalar("bow_loss", self.bow_loss)]
                                                          + self.gradient_summary)
            #生成
            with tf.name_scope("infer"):
                with tf.name_scope("group"):
                    with tf.name_scope("group_decode"):
                        def group_cond(i, group_state, gbow, groups, glens, stop):
                            return tf.cond(tf.equal(i, 0),
                                           lambda: True,
                                           lambda: tf.cond(tf.equal(tf.reduce_min(stop), 0),
                                                           lambda: tf.cond(
                                                               tf.greater_equal(i, self.config.PHVM_max_sent_cnt),
                                                               lambda: False,
                                                               lambda: True),
                                                           lambda: False))

                        def group_body(i, group_state, gbow, groups, glens, stop):
                            with tf.variable_scope("decode_group", reuse=True):
                                gout, group_state = group_decoder(gbow, group_state)

                            next_stop = tf.greater(tf.sigmoid(tf.squeeze(stop_clf(gout), axis=1)),
                                                   self.config.PHVM_stop_threshold)
                            stop += tf.cast(tf.equal(stop, 0), dtype=tf.int32) * tf.cast(next_stop, dtype=tf.int32) * (
                                        i + 1)

                            tile_gout = tf.tile(tf.expand_dims(gout, 1), [1, tf.shape(src_encoder_output)[1], 1])
                            group_fc_input = tf.concat((src_encoder_output, tile_gout), 2)
                            group_logit = tf.squeeze(group_fc_2(tf.tanh(group_fc_1(group_fc_input))), 2)


                            """
                            #选取不重复的n个最高概率topic指导文字生成
                            def select(group_prob, max_gcnt):  # 应该是根据概率进行选择
                                gid = []
                                glen = []
                                for gfid, prob in enumerate(group_prob):
                                    tmp = []
                                    max_gsid = -1
                                    max_p = -1
                                    x = []  # 2
                                    rep_record = []
                                    for gsid, p in enumerate(prob):  # 根据概率进行选择对应二元组
                                        # if p >= self.config.PHVM_group_selection_threshold:
                                        #     tmp.append([gfid, gsid])#格式就是[某个gi, 在gi中的二元组id]
                                        #     if p > max_p:
                                        #         max_gsid = gsid
                                        #         max_p = p
                                        # if len(tmp) == 0:
                                        #     tmp.append([gfid, max_gsid])
                                        if gsid not in rep_record:
                                            rep_record.append(gsid)# 如果存在不重复的gsid则加入指导生成
                                            x.append((gsid, p))
                                        else: continue
                                    x.sort(key=lambda a: a[1])  # 从小到大排序
                                    for d in x[-2:]:  # 指定每句话需要几个topic，这里每次选取n个概率最高的topic----for d in x[-n:]:
                                        tmp.append([gfid, d[0]])  # 2
                                    gid.append(tmp)
                                    glen.append(len(tmp))
                                for item in gid:
                                    if len(item) < max_gcnt:
                                        item += [[0, 0]] * (max_gcnt - len(item))  # 补齐到最长位置
                                return np.array(gid, dtype=np.int32), np.array(glen, dtype=np.int32)
                                """


                            #原始select函数
                            def select(group_prob, max_gcnt):#应该是根据概率进行选择
                                gid = []
                                glen = []
                                for gfid, prob in enumerate(group_prob):
                                    tmp = []
                                    max_gsid = -1
                                    max_p = -1
                                    for gsid, p in enumerate(prob):#根据概率进行选择对应二元组
                                        if p >= self.config.PHVM_group_selection_threshold:
                                            tmp.append([gfid, gsid])#格式就是[某个gi, 在gi中的二元组id]
                                        if p > max_p:
                                            max_gsid = gsid
                                            max_p = p
                                    if len(tmp) == 0:
                                        tmp.append([gfid, max_gsid])
                                    gid.append(tmp)
                                    glen.append(len(tmp))
                                for item in gid:
                                    if len(item) < max_gcnt:
                                        item += [[0, 0]] * (max_gcnt - len(item))#补齐到最长位置
                                return np.array(gid, dtype=np.int32), np.array(glen, dtype=np.int32)



                            src_mask = tf.sequence_mask(self.input.input_lens, tf.shape(group_logit)[1],
                                                        dtype=tf.float32)
                            group_prob = tf.sigmoid(group_logit) * src_mask
                            gid, glen = tf.py_func(select, [group_prob, tf.shape(src_encoder_output)[1]],
                                                   [tf.int32, tf.int32])#tensorflow中tensor转python中常用变量的函数
                            gid = tf.reshape(gid, (self.batch_size, -1, 2))
                            glen = tf.reshape(glen, (-1,))
                            expanded_glen = tf.expand_dims(glen, 1)
                            groups = tf.concat((groups, tf.transpose(gid[:, :, 1:], [0, 2, 1])), 1)
                            glens = tf.concat((glens, expanded_glen), 1)
                            # 利用gid进行高维矩阵（这里是对data编码以后对应的矩阵）的索引。于是group就是每个句子对应的二元组
                            group = tf.gather_nd(src_encoder_output, gid)#按照gid给的索引从src_encoder_output中抽取
                            group_mask = tf.sequence_mask(glen, tf.shape(group)[1], dtype=tf.float32)
                            expanded_group_mask = tf.expand_dims(group_mask, 2)
                            gbow = tf.reduce_sum(group * expanded_group_mask, axis=1) / tf.to_float(
                                expanded_glen)#平均池化

                            return i + 1, group_state, gbow, groups, glens, stop

                        if self.config.PHVM_rnn_type == 'lstm':
                            group_state_shape = tf.nn.rnn_cell.LSTMStateTuple(c=tf.TensorShape([None, None]),
                                                                              h=tf.TensorShape([None, None]))
                        else:
                            group_state_shape = tf.TensorShape([None, None])

                        shape_invariants = (tf.TensorShape([]), # i
                                            group_state_shape, # group_state
                                            tf.TensorShape([None, None]), # gbow
                                            tf.TensorShape([None, None, None]), # groups
                                            tf.TensorShape([None, None]), # glens
                                            tf.TensorShape([None]) # stop
                        )

                        group_state = group_init_state_fc(dec_input)
                        gbow = tf.tile(init_gbow, [self.batch_size, 1])
                        groups = tf.zeros((self.batch_size, 1, tf.shape(src_encoder_output)[1]), dtype=tf.int32)
                        glens = tf.zeros((self.batch_size, 1), dtype=tf.int32)
                        stop = tf.zeros((self.batch_size,), dtype=tf.int32)

                        _, group_state, gbow, groups, glens, stop = \
                            tf.while_loop(group_cond, group_body,
                                          loop_vars=(0, group_state, gbow, groups, glens, stop),
                                          shape_invariants=shape_invariants)

                        groups = groups[:, 1:, :]
                        glens = glens[:, 1:]

                    with tf.name_scope("group_encode"):
                        gidx, group_bow, group_mean_bow, group_embed = self.gather_group(src_encoder_output,
                                                                                         groups,
                                                                                         glens,
                                                                                         stop,
                                                                                         group_encoder)

                def infer_cond(i, plan_state, sent_state, sent_z, translations, score, len_add, cur_interval):
                    return i < tf.shape(groups)[1]

                def infer_body(i, plan_state, sent_state, sent_z, translations, score, len_add, cur_interval):

                    with tf.name_scope("group"):
                        gbow = group_mean_bow[:, i, :]
                        sent_group = group_bow[:, i, :, :]
                        sent_glen = glens[:, i]

                    # latent_decoder_input
                    if self.config.PHVM_rnn_type == 'lstm':
                        #如果是lstm，则sent_state就是一个含有h与c的tuple，h为最后一个输出hidden_state，与sent_z拼接
                        plan_input = tf.concat((sent_state.h, sent_z), 1)#lstm多一个cell state
                    else:
                        plan_input = tf.concat((sent_state, sent_z), 1)
                    sent_cond_embed, plan_state = latent_decoder(plan_input, plan_state)

                    with tf.name_scope("sent_prior_network"):
                        sent_prior_input = tf.concat((sent_cond_embed, gbow), 1)
                        sent_prior_fc = prior_fc_layer(sent_prior_input)
                        sent_prior_mu, sent_prior_logvar = tf.split(sent_prior_fc, 2, axis=1)
                        sent_z = self.sample_gaussian((self.batch_size, self.config.PHVM_sent_latent_dim),
                                                      sent_prior_mu,
                                                      sent_prior_logvar)# 指导句子生成的隐变量z采样
                        sent_z = tf.reshape(sent_z, (self.batch_size, self.config.PHVM_sent_latent_dim))

                    sent_cond_z_embed = tf.concat((sent_cond_embed, sent_z), 1)

                    with tf.name_scope("type"):
                        if self.config.PHVM_use_type_info:
                            sent_type_input = tf.concat((sent_cond_z_embed, gbow), 1)
                            sent_type_logit = type_fc_2(tf.tanh(type_fc_1(sent_type_input)))
                            sent_type_prob = tf.nn.softmax(sent_type_logit, dim=1)
                            sent_type_embed = tf.matmul(sent_type_prob, self.type_embedding)

                    #这里是sentence-decoder
                    with tf.name_scope("sent_deocde"):
                        with tf.variable_scope("sent_dec_state", reuse=True):
                            if self.config.PHVM_use_type_info:
                                sent_dec_input = tf.concat((sent_cond_z_embed, gbow, sent_type_embed), 1)
                            else:
                                sent_dec_input = tf.concat((sent_cond_z_embed, gbow), 1)
                            sent_dec_state = []
                            for _ in range(self.config.PHVM_decoder_num_layer):
                                tmp = tf.layers.dense(sent_dec_input, self.config.PHVM_decoder_dim)
                                if self.config.PHVM_rnn_type == 'lstm':
                                    tmp = tf.nn.rnn_cell.LSTMStateTuple(c=tmp, h=tmp)
                                sent_dec_state.append(tmp)
                            if self.config.PHVM_decoder_num_layer > 1:
                                sent_dec_state = tuple(sent_dec_state)
                            else:
                                sent_dec_state = sent_dec_state[0]

                        tile_glen = tf.contrib.seq2seq.tile_batch(sent_glen, multiplier=self.config.PHVM_beam_width)
                        tile_group = tf.contrib.seq2seq.tile_batch(sent_group, multiplier=self.config.PHVM_beam_width)
                        with tf.variable_scope("attention", reuse=True):
                            tile_att = tf.contrib.seq2seq.LuongAttention(self.config.PHVM_decoder_dim,
                                                                         sent_group,
                                                                         memory_sequence_length=sent_glen)
                            # tile_att = tf.contrib.seq2seq.LuongAttention(self.config.PHVM_decoder_dim,
                            #                                              tile_group,
                            #                                              memory_sequence_length=tile_glen)
                        infer_decoder = tf.contrib.seq2seq.AttentionWrapper(decoder, tile_att,
                                                                attention_layer_size=self.config.PHVM_decoder_dim)

                        tile_encoder_state = tf.contrib.seq2seq.tile_batch(sent_dec_state,
                                                                           multiplier=self.config.PHVM_beam_width)
                        # sent_decoder初始状态
                        decoder_initial_state = infer_decoder.zero_state(
                            self.batch_size, dtype=tf.float32).clone(cell_state=sent_dec_state)
                        # decoder_initial_state = infer_decoder.zero_state(
                        #     self.batch_size * self.config.PHVM_beam_width,
                        #     dtype=tf.float32).clone(cell_state=tile_encoder_state)
                        #beamsearch概率
                        self.beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                            cell=infer_decoder,
                            embedding=self.word_embedding,
                            start_tokens=tf.tile([self.start_token], [self.batch_size]),
                            end_token=self.end_token,
                            initial_state=decoder_initial_state,
                            beam_width=self.config.PHVM_beam_width,
                            output_layer=projection,
                            length_penalty_weight=0.0
                        )


                        if self.hide_strategy == 'RS':
                            self.decoder_helper = PHVM_RSEmbeddingHelper.SampleEmbeddingHelper(embedding=self.word_embedding,
                                                                                       start_tokens=tf.tile([self.start_token], [self.batch_size]), end_token=self.end_token,
                                                                                       tuple_ids=self.input.val_word, no_hide=self.no_hide, k=5, sample=True, vocab_size=self.truncated_vocab_size,
                                                                                    add=len_add, bitfile=self.bitfile, bit_per_word=config.bit_per_word, indices=self.indices)
                            self.inference_decoder = RS_BasicDecoder(cell=infer_decoder,
                                                                  helper=self.decoder_helper,
                                                                  initial_state=decoder_initial_state,
                                                                  output_layer=projection)
                        elif self.hide_strategy == 'AC':
                            self.decoder_helper = AC_SampleEmbeddingHelper.SampleEmbeddingHelper(embedding=self.word_embedding,
                                                                                       start_tokens=tf.tile([self.start_token], [self.batch_size]), end_token=self.end_token,
                                                                                       tuple_ids=self.input.val_word, k=5, add=len_add, bitfile=self.bitfile, precision=config.AC_precision,
                                                                                       indices=self.AC_indices, bpw=config.bit_per_word, STRATEGY=self.STRATAGY, vocab_size=self.truncated_vocab_size)
                            self.inference_decoder = AC_BasicDecoder(cell=infer_decoder, helper=self.decoder_helper,
                                                                     initial_state=decoder_initial_state,
                                                                     output_layer=projection)
                        elif self.hide_strategy == 'no':
                            self.decoder_helper = PHVM_SampleEmbeddingHelper.SampleEmbeddingHelper(
                                embedding=self.word_embedding, k=5, kww=False, vocab_size=self.truncated_vocab_size,tuple_ids=self.input.val_word,
                                start_tokens=tf.tile([self.start_token], [self.batch_size]), end_token=self.end_token)

                            self.inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=infer_decoder, helper=self.decoder_helper,
                                                                     initial_state=decoder_initial_state,
                                                                     output_layer=projection)



                        """
                        self.decoder_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                            embedding=self.word_embedding,
                            start_tokens=tf.tile([self.start_token], [self.batch_size]), end_token=self.end_token, softmax_temperature=1.0)
                        
                        self.decoder_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                            embedding=self.word_embedding,
                            start_tokens=tf.tile([self.start_token], [self.batch_size]), end_token=self.end_token)
                        """
                        # self.inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=infer_decoder, helper=self.decoder_helper, initial_state=decoder_initial_state,
                        #                                                      output_layer=projection)
                        with tf.variable_scope("dynamic_decoding", reuse=True):
                            # beamsearch decoder
                            # fout, fstate, flen = tf.contrib.seq2seq.dynamic_decode(self.inference_decoder,
                            #                                  output_time_major=False,
                            #                             maximum_iterations=self.config.PHVM_maximum_iterations)
                            # basic decoder
                            # fout, fstate, flen = tf.contrib.seq2seq.dynamic_decode(self.inference_decoder,
                            #                                                        impute_finished=False, swap_memory=False,
                            #                                                        maximum_iterations=self.config.PHVM_maximum_iterations)
                            if self.hide_strategy == 'RS':
                                fout, fstate, flen, fbnum = RS_dynamic_decode(self.inference_decoder,
                                                                                   impute_finished=True,
                                                                                   maximum_iterations=self.config.PHVM_maximum_iterations)
                                finterval = cur_interval
                            elif self.hide_strategy == 'AC':
                                fout, fstate, flen, fbnum, finterval = AC_dynamic_decode(decoder=self.inference_decoder,
                                                                                     init_interval=cur_interval, impute_finished=True,
                                                                                   maximum_iterations=self.config.PHVM_maximum_iterations)
                            elif self.hide_strategy == 'no':
                                fout, fstate, flen = tf.contrib.seq2seq.dynamic_decode(self.inference_decoder,
                                                                                   impute_finished=True,
                                                                                   maximum_iterations=self.config.PHVM_maximum_iterations)
                                fbnum = 0
                                finterval = cur_interval

                            sent_output = fout.sample_id
                            sent_output = sent_output * tf.to_int32(tf.greater_equal(sent_output, 0))# Returns the truth value of (x >= y) element-wise.
                            len_add += fbnum
                            dist = self.config.PHVM_maximum_iterations - tf.shape(sent_output)[1] # 此刻生成句子和最长句子之间的差距
                            padded_sent_output = tf.cond(tf.greater(dist, 0),
                                                         lambda: tf.concat(
                                                             (sent_output,
                                                              tf.zeros((self.batch_size, dist), dtype=tf.int32)), 1),#tf.zeros((self.batch_size, dist), dtype=tf.int32)), 1)
                                                         lambda: sent_output)
                            padded_score_output = tf.cond(tf.greater(dist, 0),
                                                         lambda: tf.concat(
                                                             (fout.rnn_output,
                                                              tf.zeros((self.batch_size, dist, self.tgt_vocab_size), dtype=tf.float32)), 1),
                                                         lambda: fout.rnn_output)
                            translations = tf.concat((translations, tf.expand_dims(padded_sent_output, 1)), 1)
                            score = tf.concat((score, tf.expand_dims(padded_score_output,1)), 1)
                            pad_output = tf.concat((tf.zeros((self.batch_size, 1), dtype=tf.int32), sent_output), 1)
                            sent_lens = tf.argmax(tf.cast(tf.equal(pad_output, 1), dtype=tf.int32), 1,
                                                  output_type=tf.int32)
                            sent_lens = sent_lens - 1 + tf.to_int32(tf.equal(sent_lens, 0)) * (tf.shape(sent_output)[1] + 1)

                        with tf.variable_scope("attention", reuse=True):
                            att = tf.contrib.seq2seq.LuongAttention(self.config.PHVM_decoder_dim,
                                                                    sent_group,
                                                                    memory_sequence_length=sent_glen)
                        infer_encoder = tf.contrib.seq2seq.AttentionWrapper(decoder, att,
                                                            attention_layer_size=self.config.PHVM_decoder_dim)
                        sent_input = tf.nn.embedding_lookup(self.word_embedding, sent_output)
                        encoder_state = infer_encoder.zero_state(self.batch_size, dtype=tf.float32).clone(
                            cell_state=sent_dec_state)
                        helper = tf.contrib.seq2seq.TrainingHelper(sent_input, sent_lens, time_major=False)
                        basic_decoder = tf.contrib.seq2seq.BasicDecoder(infer_encoder, helper, encoder_state,
                                                                        output_layer=projection)
                        with tf.variable_scope("dynamic_decoding", reuse=True):
                            fout0, fstate, flens = tf.contrib.seq2seq.dynamic_decode(basic_decoder,
                                                                                    impute_finished=True)

                        with tf.variable_scope("sent_state_update", reuse=True):
                            sent_state = fstate.cell_state[self.config.PHVM_decoder_num_layer - 1]

                    return i + 1, plan_state, sent_state, sent_z, translations, score, len_add, finterval

                if self.config.PHVM_rnn_type == 'lstm':
                    plan_state_shape = tf.nn.rnn_cell.LSTMStateTuple(c=tf.TensorShape([None, None]),
                                                                     h=tf.TensorShape([None, None]))
                else:
                    plan_state_shape = tf.TensorShape([None, None])
                sent_state_shape = plan_state_shape
                shape_invariants = (tf.TensorShape([]), # i
                                    plan_state_shape, # plan_state
                                    sent_state_shape, # sent_state
                                    tf.TensorShape([None, None]), # sent_z

                                    tf.TensorShape([None, None, None]), # translations

                                    tf.TensorShape([None, None, None, None]), # score

                                    tf.TensorShape([]),  # len_add

                                    tf.TensorShape([2]), # AC interval
                                    )
                plan_state = plan_init_state_fc(tf.concat((dec_input, group_embed), 1))
                sent_state = decoder.zero_state(self.batch_size, dtype=tf.float32)[
                    self.config.PHVM_decoder_num_layer - 1]
                sent_z = tf.zeros(shape=(self.batch_size, self.config.PHVM_sent_latent_dim), dtype=tf.float32)
                translations = tf.zeros((self.batch_size, 1, self.config.PHVM_maximum_iterations), dtype=tf.int32)
                # score = tf.zeros(shape=(self.batch_size, self.val_vocab_size, self.config.PHVM_beam_width))#最后一维是beamwidth
                score = tf.zeros(shape=(self.batch_size, 1, self.config.PHVM_maximum_iterations, self.tgt_vocab_size))
                init_interval = tf.constant([0, 2**config.AC_precision], dtype=tf.float32) # float32 is more compatible with computation

                cnt, plan_state, sent_state, sent_z, translations, score, len_add, final_interval = \
                    tf.while_loop(infer_cond,
                                  infer_body,
                                  loop_vars=(0, plan_state, sent_state, sent_z, translations, score, 0, init_interval),
                                  shape_invariants=shape_invariants)



                self.stop = stop + tf.cast(tf.equal(stop, 0), dtype=tf.int32) * self.config.PHVM_max_sent_cnt
                self.groups = groups
                self.glens = glens
                self.translations = translations[:, 1:, :]
                self.score = score[:, 1:, :]
                self.cnt = cnt
                self.info_len = len_add
                self.interval = final_interval



    def get_global_step(self):
        return self.sess.run(self.global_step)

    def train(self, batch_input):
        feed_dict = {key: val for key, val in zip(self.input, batch_input)}
        feed_dict[self.keep_prob] = 1 - self.config.PHVM_dropout
        feed_dict[self.train_flag] = True
        _, global_step, train_loss, summary = self.sess.run(
            (self.update, self.global_step, self.train_loss, self.train_summary), feed_dict=feed_dict)

        return global_step, train_loss, summary

    def eval(self, batch_input):
        feed_dict = {key: val for key, val in zip(self.input, batch_input)}
        feed_dict[self.keep_prob] = 1
        feed_dict[self.train_flag] = True
        global_step, loss = self.sess.run((self.global_step, self.elbo_loss), feed_dict=feed_dict)
        return global_step, loss


    def infer(self, batch_input, bitfile):
        feed_dict = {self.input.key_input: batch_input.key_input,
                     self.input.val_input: batch_input.val_input,
                     self.input.val_word: batch_input.val_word[0][2:],
                     self.input.input_lens: batch_input.input_lens,
                     self.input.category: batch_input.category,

                     self.input.text: np.array([[0]] * len(batch_input.category)),#长度为len(input.category)的[[0], [0], ..., [0]]
                     self.input.slens: np.array([1] * len(batch_input.category)),#长度为len(input.category)的[1, 1, ..., 1]
                     self.bitfile: bitfile,
                     self.no_hide: [327, 325], # 句号，<UNK>，逗号不参与隐藏
                     }
        feed_dict[self.keep_prob] = 1
        feed_dict[self.train_flag] = False
        #groups可能是句子对应的二元组描述

        stop, groups, glens, translations, score, cnt, bit_len, final_interval = \
            self.sess.run((self.stop, self.groups, self.glens, self.translations, self.score,
                           self.cnt, self.info_len, self.interval), feed_dict=feed_dict)
        #输出具体指导每句话生成的二元组组合
        # tota = model_utils.extraction(groups.tolist(), model_utils.load_data(config.test_file))
        # 测试输出
        # print('此时长度为：{}'.format(len(score[0][0])))
        # for i in range(len(score[0][0])):
        #     todivide = 0
        #
        #     for k in range(len(score[0][0[i])):
        #         score[0][i][k] = math.exp(score[0][0][i][k])
        #         todivide+=score[0][0[i][k]
        #
        #     for j in range(len(score[0][i])):
        #         score[0][i][j] = -1*(score[0][i][j]/todivide)
        #     score[0][i].sort()

        # print('此时的val_word是{}'.format(batch_input.val_word[0][2:]))
        # print('此时的val_input是{}'.format(batch_input.val_input))
        # print('此时的groups为{}'.format(groups))
        # print('此时的glens为{}'.format(glens))
        # print('此时的tranlation{}，长度为{}'.format(translations, len(translations[0])))
        # print('此时的stop{}'.format(stop))
        # print('此时的bit_len {}'.format(bit_len))
        # print('此时的final_interval为：{}'.format(final_interval))
        return self._agg_group(stop, translations), bit_len

    def _agg_group(self, stop, text):
        translation = []
        for gcnt, sent in zip(stop, text):
            sent = sent[:gcnt, :]
            desc = []
            for segId, seg in enumerate(sent):
                for wid in seg:
                    if wid == self.end_token:
                        break
                    elif wid == self.start_token:
                        continue
                    else:
                        desc.append(wid)
            translation.append(desc)
        max_len = 0
        for sent in translation:
            max_len = max(max_len, len(sent))
        for i, sent in enumerate(translation):
            translation[i] = [sent + [self.end_token] * (max_len - len(sent))]#句子长短不一，为了保证长度都一样，拿end_token进行补全操作，补至与最长的句子长度一致
        return translation
