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
               indices, vocab_size, SD, softmax_temperature=None, seed=None):
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
    self.k = k + int(precision / 2)
    self.precision = precision
    # self.STRATEGY = STRATEGY
    self.divide_num = 2**precision
    self.indices = indices # [[0 ,2, 4, ..], [1, 3, 5...]] for 1 AC precision
    self._add = add
    self.bitfile = bitfile
    self.truncated_vocab_size = vocab_size
    self.thr = 0.5
    self.SD = SD

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
      return tf.reduce_sum(tf.cast(tf.reverse(tensor=binary_string, axis=[0]), dtype=tf.int32) * 2 ** tf.range(
          tf.cast(tf.size(binary_string), dtype=tf.int32)))
  def tf_int2bit(self, int_num, pow_length):
      pow_constant = tf.constant([2**ii for ii in range(pow_length, -1, -1)])
      return tf.reshape(tf.mod(tf.to_int32(int_num // pow_constant), 2), [-1])

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
      _sorted_probs = tf.nn.softmax(sorted_logits) # convert to probability
      cur_interval_original = cur_interval

      if self.SD is False:
          sorted_probs = _sorted_probs
          sorted_ind = _sorted_ind
      else:
          prob_group = [tf.gather(_sorted_probs, ii) for ii in self.indices]
          id_group = [tf.gather(_sorted_ind, ii) for ii in self.indices]
          select_index = tf.mod(self.tf_bit2int(tf.reverse(message_bit), [0]), self.precision)
          _sorted_probs = tf.gather(prob_group, select_index)
          sorted_probs = _sorted_probs / tf.reduce_sum(_sorted_probs)
          sorted_ind = tf.gather(id_group, select_index)


      cur_int_range = cur_interval[1] - cur_interval[0]
      cur_thr = 1 / cur_int_range

      k = tf.maximum(2, tf.argmax(tf.cast(tf.equal(sorted_probs < cur_thr, True), tf.int32)))
      sorted_probs, sorted_ind = control_flow_ops.cond(cur_thr > sorted_probs[-1],
                                                       lambda: self.cond0_true(sorted_probs, sorted_ind, k),
                                                       lambda: self.cond0_false(sorted_probs, sorted_ind, k))

      prob = sorted_probs / tf.reduce_sum(sorted_probs)
      prob *= cur_int_range
      prob = tf.round(prob)

      cum_prob = tf.cumsum(prob)
      judge_bool_cum = cum_prob>cur_int_range
      cum_prob = control_flow_ops.cond(tf.count_nonzero(judge_bool_cum),
                                       lambda: cum_prob[:tf.argmax(tf.cast(tf.equal(judge_bool_cum, True), tf.int32))],
                                       lambda: cum_prob)
      cum_prob += cur_int_range - cum_prob[-1]
      cum_prob += cur_interval[0]

      message_idx = self.tf_bit2int(tf.reverse(message_bit, [0]))
      selection = tf.argmax(tf.cast(tf.equal(cum_prob>message_idx), tf.int32))

      new_int_bottom = control_flow_ops.cond(selection>0, lambda: cum_prob[selection - 1], lambda: cur_interval[0])
      new_int_top = cum_prob[selection]

      new_int_bottom_bits_enc = self.tf_int2bit(new_int_bottom, self.precision)
      new_int_top_bits_enc = self.tf_int2bit(new_int_top - 1, self.precision)

      num_bits_encoded = tf.argmax(tf.cast(tf.not_equal(new_int_bottom_bits_enc, new_int_top_bits_enc), tf.int32))

      new_int_bottom_bits = tf.concat((new_int_bottom_bits_enc[:num_bits_encoded], tf.zeros(shape=(num_bits_encoded), dtypes=tf.int32)), 0)
      new_int_top_bits = tf.concat((new_int_top_bits_enc[num_bits_encoded:], tf.ones(shape=(num_bits_encoded), dtypes=tf.int32)), 0)

      cur_interval[0] = self.tf_bit2int(tf.reverse(new_int_bottom_bits, [0]))
      cur_interval[1] = self.tf_bit2int(tf.reverse(new_int_top_bits, [0])) + 1



      # tuple priority judgement
      broadcast_equal = tf.equal(tf.reshape(self.tuple_ids, (-1, 1)), sorted_ind[:self.k])  # 自动对齐
      nz_cnt = tf.count_nonzero(broadcast_equal, axis=0)
      tuple_bool = tf.equal(nz_cnt, 1)

      hidden_sign = tf.greater(sorted_logits[0] - sorted_logits[1], self.thr) # priority3: probability threshold judgement

      judge_cnt = tf.reduce_sum(tf.cast(tuple_bool, tf.float32))  # judge if key words in topk prob indicies
      # priority2: probability threshold judgement
      _sample_ids = control_flow_ops.cond(hidden_sign, lambda: sorted_ind[0], lambda: sorted_ind[selection])

      # priority1: key tuples judegement
      # shape of sample_ids: [batch_size, 1]
      sample_ids = control_flow_ops.cond(tf.greater(judge_cnt, 0),
                                         lambda: sorted_ind[tf.argmax(tf.cast(tf.equal(tuple_bool, True), tf.int32))],
                                         lambda: _sample_ids)
      hidden_sign = control_flow_ops.cond(tf.greater(judge_cnt, 0),
                                          lambda: True, lambda: hidden_sign)  # topk include key tuples = no hide = True

      cur_interval_final = control_flow_ops.cond(hidden_sign, lambda: cur_interval_original, lambda: cur_interval)
      num_bits_encoded_final = control_flow_ops.cond(hidden_sign, lambda: 0, lambda: num_bits_encoded)

      # 返回某步选取出来的
      return tf.expand_dims(sample_ids, 0), num_bits_encoded_final, cur_interval_final