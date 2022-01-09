#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@file:   RS_EmbeddingHelper.py
@author: Haoqin
@date:   2020/09/01

"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops
import numpy as np
import math

class SampleEmbeddingHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):
  """A helper for use during inference.
  Uses sampling (from a distribution) instead of argmax and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, start_tokens, end_token, tuple_ids, k,
               vocab_size, kww=True, sample=True, softmax_temperature=None, seed=None):
      # batch size is 1 to make the judge every step
    """
      :param embedding:
      :param start_tokens:
      :param end_token:
      :param tuple_ids:
      :param no_hide:
      :param k:
      :param add:
      :param bitfile:
      :param bit_per_word:
      :param indices:
      :param vocab_size:
      :param sample:
      :param softmax_temperature:
      :param seed:
      """
    super(SampleEmbeddingHelper, self).__init__(embedding, start_tokens, end_token)
    self._softmax_temperature = softmax_temperature
    self._seed = seed
    self.tuple_ids = tuple_ids
    self.k = k
    self.no_hide_k = 1 # specified symbols in topk prob then hide = False
    self.sample_sign = sample
    self.truncated_vocab_size = vocab_size
    self.kww = kww
    self.thr = 0.8

  def toargmax(self, outputs):
      # Greedy Embedding output
      return math_ops.argmax(outputs, axis=-1, output_type=dtypes.int32)


  def topk_topic(self, outputs, tuple_ids, k):
      """
      :param outputs: prob output of RNN cell
      :return: topk_output_indices: list of bool tensor true if in key tuples
      :return: tuple_bool: bool tensor true for key word in topk prob of outputs
      """

      '''
                broadcast_equal = (self.tuple_ids[:, None] == topk_output_indices[0])  #[true, false, ...] (m * k-dim)
                nz_cnt = tf.count_nonzero(broadcast_equal, axis = 1) (k-dim)
                tuple_bool = (nz_cnt == 1)
      
      '''
      _, topk_output_indices = tf.nn.top_k(outputs, k) # 降序排列  (1, k)
      broadcast_equal = tf.equal(tf.reshape(tuple_ids, (-1, 1)), topk_output_indices) # 自动对齐
      nz_cnt = tf.count_nonzero(broadcast_equal, axis = 0)
      tuple_bool = tf.equal(nz_cnt, 1)
      return topk_output_indices, tuple_bool


  def sample(self, time, outputs, state, name=None):
      """
      :param time:
      :param outputs:
      :param state:
      :param bit_num:
      :param name:
      :return:
      """
      # todo: 1. 高概率词直接输出不进行采样 2.truncated sampling 设置较小的truncated size 3.输出对应句子实际的bpw 4.随着bpw的增加，增加topk的k值
      # todo: 5. 动态变化sigma的值，使得对应分布差不多平均
      del time, state  # unused by sample_fn

      if not isinstance(outputs, ops.Tensor):
          raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                          type(outputs))

      if self._softmax_temperature is None:
          logits = outputs
      else:
          logits = outputs / self._softmax_temperature


      sorted_logits, sorted_ind = tf.nn.top_k(logits[0], self.truncated_vocab_size)
      sorted_probs = tf.nn.softmax(sorted_logits)

      # tuple priority judgement
      broadcast_equal = tf.equal(tf.reshape(self.tuple_ids, (-1, 1)), sorted_ind[:self.k])  # 自动对齐
      nz_cnt = tf.count_nonzero(broadcast_equal, axis=0)
      tuple_bool = tf.equal(nz_cnt, 1)
      hidden_sign = tf.greater(sorted_probs[0] - sorted_probs[1], self.thr)

      # hidden_sign = tf.greater(sorted_logits[0] - logits_div[0], self.thr - tf.exp(self.bit_per_word / 4) * 0.1)
      if self.sample_sign is True:
          # conditional Categorical sample method
          sample_id_sampler = categorical.Categorical(logits=sorted_logits)  # batch_size = 1
          sample_id_init = sorted_ind[sample_id_sampler.sample(seed=self._seed)]
      else:
          # conditional Greedy sample method
          sample_id_init = sorted_ind[0]

      judge_cnt = tf.reduce_sum(tf.cast(tuple_bool, tf.float32))  #  judge if key words in topk prob indicies
      # priority2: probability threshold judgement
      _sample_ids = control_flow_ops.cond(hidden_sign, lambda: sorted_ind[0], lambda: sample_id_init)

      # priority1: key tuples judegement
      # shape of sample_ids: [batch_size, 1]
      if self.kww:
        sample_ids = control_flow_ops.cond(tf.greater(judge_cnt, 0),
                                         lambda: sorted_ind[tf.argmax(tf.cast(tf.equal(tuple_bool, True), tf.int32))],
                                         lambda: _sample_ids)
      else:
          sample_ids = _sample_ids
      # predict_ids pass to next step
      return tf.expand_dims(sample_ids, 0)
