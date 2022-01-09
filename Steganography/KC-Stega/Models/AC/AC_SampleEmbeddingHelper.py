#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@file:   AC_EmbeddingHelper.py
@author: Haoqin
@date:   2020/12/10

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

  def __init__(self, embedding, start_tokens, end_token, tuple_ids, k, add, bitfile, precision,
               indices, vocab_size, bpw, STRATEGY='', softmax_temperature=None, seed=None):
      # batch size is 1 to make the judge every step
    """Initializer.
    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      softmax_temperature: (Optional) `float32` scalar, value to divide the
        logits by before computing the softmax. Larger values (above 1.0) result
        in more random samples, while smaller values push the sampling
        distribution towards the argmax. Must be strictly greater than 0.
        Defaults to 1.0.
      seed: (Optional) The sampling seed.
    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """
    super(SampleEmbeddingHelper, self).__init__(embedding, start_tokens, end_token)
    self._softmax_temperature = softmax_temperature
    self._seed = seed
    self.tuple_ids = tuple_ids
    self.k = k
    self.precision = precision
    # self.STRATEGY = STRATEGY
    self.divide_num = 2**precision
    self.indices = indices # [[0 ,2, 4, ..], [1, 3, 5...]] for 1 AC precision
    self._add = add
    self.bitfile = bitfile
    self.truncated_vocab_size = vocab_size
    self.thr = 0.6
    self.STRATEGY = STRATEGY
    self.bpw = bpw
    self.augument_rate = 90

  def toargmax(self, outputs):
      # Greedy Embedding output
      return math_ops.argmax(outputs, axis=-1, output_type=dtypes.int32)

  def topk_topic(self, outputs, tuple_ids, k):
      """
      :param outputs: prob output of RNN cell
      :return: topk_output_indices: list of bool tensor true if in key tuples
      :return: tuple_bool: bool tensor true for key word in topk prob of outputs
      """
      _, topk_output_indices = tf.nn.top_k(outputs, k) # 降序排列  (1, k)
      broadcast_equal = tf.equal(tf.reshape(tuple_ids, (-1, 1)), topk_output_indices)# 自动对齐
      nz_cnt = tf.count_nonzero(broadcast_equal, axis = 0)
      tuple_bool = tf.equal(nz_cnt, 1)
      return topk_output_indices, tuple_bool

  """
  def strategy_process(self, STRATEGY, logits):
      sorted_logits, sorted_ind = tf.nn.top_k(logits[0], self.truncated_vocab_size)
      if STRATEGY=='':
          logits = sorted_logits
          ind = sorted_ind
      elif STRATEGY=='sample':
          sample_id_sampler = categorical.Categorical(logits=sorted_logits)

      elif STRATEGY=='threshold':

      return logits, ind
      """


  def tf_bit2int(self, binary_string):
      """
      :param binary_string: bit list
      :return: corrosponding decimal number
      """
      # return dtype=tf.float32 for easy comparison
      return tf.reduce_sum(tf.cast(tf.reverse(binary_string, [0]),
                                   dtype=tf.float32) * 2 ** tf.range(tf.cast(tf.size(binary_string), dtype=tf.float32)))

  def tf_int2bit(self, int_num, pow_length):
      pow_constant = tf.constant([2**ii for ii in range(pow_length - 1, -1, -1)])
      return tf.reshape(tf.mod(tf.to_int32(int_num // pow_constant), 2), [-1])

  def magnify_probs(self, probs, portion_control=None):
      """
      :param logits: RNN output
      :param portion_control: param tune for categorical sampling
      :param temperature:
      :return: tuned and normaized logits
      """
      probs /= tf.reduce_sum(probs)
      # logits = logits / tf.reduce_sum(logits)
      if portion_control is None:
          raw_portion = probs[0] / probs[1]
          x1_control = tf.log(raw_portion)/ (raw_portion - 1)
          portion_control = (x1_control * self.augument_rate)/ probs[1]
      probs = tf.exp(tf.multiply(probs, portion_control)) # todo: tune the portion_control
      return probs / tf.reduce_sum(probs)

  # control_flow_ops.cond() 函数参数
  def cond0_true(self, prob, indi, k):
      return prob[:k], indi[:k]
  def cond0_false(self, prob, indi, k):
      return prob, indi


  def sample(self, time, outputs, state, bit_num, cur_interval, name=None):
      del time, state  # unused by sample_fn
      bit_num += self._add
      message_bit = self.bitfile[bit_num: bit_num + self.precision]


      if not isinstance(outputs, ops.Tensor):
          raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                          type(outputs))

      if self._softmax_temperature is None:
          logits = outputs
      else:
          logits = outputs / self._softmax_temperature


      sorted_logits, _sorted_ind = tf.nn.top_k(logits[0], self.truncated_vocab_size)

      # tuple priority judgement
      broadcast_equal = tf.equal(tf.reshape(self.tuple_ids, (-1, 1)), _sorted_ind[:self.k])  # 自动对齐
      nz_cnt = tf.count_nonzero(broadcast_equal, axis=0)
      tuple_bool = tf.equal(nz_cnt, 1)

      _sorted_probs = tf.nn.softmax(sorted_logits) # convert to probability
      hidden_sign = tf.greater(_sorted_probs[0] - _sorted_probs[1], self.thr)  # priority3: probability threshold judgement
      cur_interval_original = cur_interval

      if self.STRATEGY == '':
          sorted_probs = _sorted_probs
          sorted_ind = _sorted_ind
      elif self.STRATEGY == 'topk':
          sorted_probs = _sorted_probs[:2 ** self.bpw]
          sorted_ind = _sorted_ind[:2 ** self.bpw]
      elif self.STRATEGY == 'sample':
          magnified_sorted_probs = self.magnify_probs(_sorted_probs)
          noise_z = -tf.log(-tf.log(tf.random_uniform(tf.shape(magnified_sorted_probs), 0, 1)))
          _, indices_toindex = tf.nn.top_k(tf.log(magnified_sorted_probs) + noise_z, 2 ** self.bpw)
          sorted_probs = tf.gather(_sorted_probs, indices_toindex)
          sorted_ind = tf.gather(_sorted_ind, indices_toindex)
      else:
          prob_group = [tf.gather(_sorted_probs, ii) for ii in self.indices]
          id_group = [tf.gather(_sorted_ind, ii) for ii in self.indices]
          select_index = tf.to_int32(tf.mod(self.tf_bit2int(message_bit), self.precision))
          _sorted_probs = tf.gather(prob_group, select_index)
          sorted_probs = _sorted_probs / tf.reduce_sum(_sorted_probs)
          sorted_ind = tf.gather(id_group, select_index)


      cur_int_range = cur_interval[1] - cur_interval[0]
      cur_thr = 1 / cur_int_range

      k = tf.maximum(2, tf.to_int32(tf.argmax(tf.cast(tf.equal(sorted_probs < cur_thr, True), tf.int32))))
      sorted_probs, sorted_ind = control_flow_ops.cond(cur_thr > sorted_probs[-1],
                                                       lambda: self.cond0_true(sorted_probs, sorted_ind, k),
                                                       lambda: self.cond0_false(sorted_probs, sorted_ind, k))

      prob = sorted_probs / tf.reduce_sum(sorted_probs)
      prob *= cur_int_range
      prob = tf.round(prob)

      cum_prob = tf.cumsum(prob)
      over_fill_index = (cum_prob > cur_int_range)
      cum_prob = control_flow_ops.cond(tf.count_nonzero(over_fill_index) > 0,
                                       lambda: cum_prob[:tf.argmax(tf.cast(tf.equal(over_fill_index, True), tf.int32))],
                                       lambda: cum_prob)
      cum_prob += cur_int_range - cum_prob[-1]
      cum_prob += cur_interval[0]

      message_idx = self.tf_bit2int(message_bit)
      selection = tf.argmax(tf.cast(tf.equal(cum_prob > message_idx, True), tf.int32))

      new_int_bottom = tf.to_int32(control_flow_ops.cond(selection > 0,
                                                         lambda: cum_prob[selection - 1], lambda: cur_interval[0]))
      new_int_top = tf.to_int32(cum_prob[selection])

      new_int_bottom_bits_enc = self.tf_int2bit(new_int_bottom, self.precision)
      new_int_top_bits_enc = self.tf_int2bit(new_int_top - 1, self.precision)

      num_bits_encoded = tf.to_int32(tf.argmax(tf.cast(tf.not_equal(new_int_bottom_bits_enc, new_int_top_bits_enc), tf.int32)))
      # print(num_bits_encoded)

      new_int_bottom_bits = tf.concat([new_int_bottom_bits_enc[num_bits_encoded:],
                                                                    tf.zeros(shape=(num_bits_encoded,), dtype=tf.int32)], 0)
      new_int_top_bits = tf.concat([new_int_top_bits_enc[num_bits_encoded:],
                                                                 tf.ones(shape=(num_bits_encoded,), dtype=tf.int32)], 0)
      cur_interval = tf.concat(([self.tf_bit2int(new_int_bottom_bits)], [self.tf_bit2int(new_int_top_bits) + 1]), 0)



      # priority2: probability threshold judgement
      _sample_ids = control_flow_ops.cond(hidden_sign, lambda: _sorted_ind[0], lambda: sorted_ind[selection])

      # priority1: key tuples judegement
      # shape of sample_ids: [batch_size, 1]
      judge_cnt = tf.reduce_sum(tf.cast(tuple_bool, tf.float32))  # judge if key words in topk prob indicies
      sample_ids = control_flow_ops.cond(tf.greater(judge_cnt, 0),
                                         # original sorted ids (without truncation or SD operation)
                                         lambda: _sorted_ind[tf.argmax(tf.cast(tf.equal(tuple_bool, True), tf.int32))],
                                         lambda: _sample_ids)
      hidden_sign = control_flow_ops.cond(tf.greater(judge_cnt, 0), lambda: True, lambda: hidden_sign)  # topk include key tuples = no hide = True

      # hidden_sign=True: final out interval = original interval
      cur_interval_final = control_flow_ops.cond(hidden_sign, lambda: cur_interval_original, lambda: cur_interval)
      num_bits_encoded_final = control_flow_ops.cond(hidden_sign, lambda: 0, lambda: num_bits_encoded)

      # predict_ids.shape = [batch_size=1, id_number=1]             cur_interval_final pass to next AC stage
      return tf.expand_dims(sample_ids, 0), num_bits_encoded_final, cur_interval_final