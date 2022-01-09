#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@file:   one_gen.py
@author: Haoqin
@date:   2020/11/30

"""

import os
import sys
import numpy as np
import json
import argparse
import Config
import pickle
import time
from Models.model_utils import bits2int

def import_lib():
	global Dataset, utils, tf, device_lib, PHVM, Dataset, model_utils
	import tensorflow as tf
	from tensorflow.python.client import device_lib
	import utils

	import Dataset

	from Models import PHVM
	from Models import model_utils

def dump(texts, filename):
	file = open(filename, "w")
	for inst in texts:
		lst = []
		for sent in inst:
			sent = " ".join(sent)
			lst.append({'desc': sent})
		file.write(json.dumps(lst, ensure_ascii=False) + "\n")
	file.close()

#改变测试数据集就可以指定整个广告文案生成包含哪些data的句子，这里怎么控制某一句话一定包含某个二元组data
#train 的时候还是要用后验分布，即把结果输入进行训练
def infer(model, dataset, data, bitfile):#test时输出文本
	"""
	:param model:
	:param dataset:
	:param data:
	:return:
	"""
	config = Config.config
	brand_set = pickle.load(open(config.brand_set_file, "rb"))
	vocab = dataset.vocab
	batch = dataset.get_batch(data)
	res = []
	res_id = []
	bit_len = []
	while True:
		try:
			# print('此时的batch_id是{}'.format(batch_id))
			batchInput = dataset.next_batch(batch)#一个batch地输入
			output, _bit_len = model.infer(batchInput, bitfile)#output应该是输出[[], [], [], ...]，每个[]由wid组成，每个[]为一个sentence
			bit_len.append(_bit_len)
			_output = []
			_output_id = []
			for inst_id, inst in enumerate(output):
				sents = []
				sents_id = []
				dup = set()
				#beam是一个句子中字对应id的集合
				for beam in inst:
					sent = []
					sent_id = []
					for wid in beam:
						if wid == dataset.vocab.end_token:#到了vocab尾部则结束
							break
						elif wid == dataset.vocab.start_token:
							continue
						#句子中每个字的生成
						# sent.append(vocab.id2word[wid] if vocab.id2word[wid] not in brand_set else "BRAND")#生成句子不是品牌名称
						sent_id.append(wid)
					if str(sent) not in dup:
						dup.add(str(sent))
						sents.append(sent)
						sents_id.append(sent_id)
				_output.append(sents)
				_output_id.append(sents_id)
			res.extend(_output)
			res_id.append(_output_id)
		except tf.errors.OutOfRangeError:
			break
	return res, res_id, bit_len

def evaluate(model, dataset, data):
	batch = dataset.get_batch(data)
	tot_loss = 0
	tot_cnt = 0
	while True:
		try:
			batchInput = dataset.next_batch(batch)
			global_step, loss = model.eval(batchInput)
			slens = batchInput.slens
			tot_cnt += len(slens)
			tot_loss += loss * len(slens)
		except tf.errors.OutOfRangeError:
			break
	return tot_loss / tot_cnt

def _train(model_name, model, dataset, summary_writer, init):
	best_loss = 1e20
	batch = dataset.get_batch(dataset.train)
	epoch = init['epoch']
	worse_step = init['worse_step']
	logger.info("epoch {}".format(epoch))
	if model.get_global_step() > config.num_training_step or worse_step > model.early_stopping:
		return
	while True:
		try:
			batchInput = dataset.next_batch(batch)
			global_step, loss, train_summary = model.train(batchInput)

			if global_step % config.steps_per_stat == 0:
				summary_writer.add_summary(train_summary, global_step)
				summary_writer.flush()
				logger.info("{} step : {:.5f}".format(global_step, loss))
		except tf.errors.OutOfRangeError:
			eval_loss = evaluate(model, dataset, dataset.dev)
			utils.add_summary(summary_writer, global_step, "dev_loss", eval_loss)
			logger.info("dev loss : {:.5f}".format(eval_loss))

			if eval_loss < best_loss:
				worse_step = 0
				best_loss = eval_loss
				prefix = config.checkpoint_dir + "/" + model_name + config.best_model_dir
				model.best_saver.save(model.sess, prefix + "/best_{}".format(epoch), global_step=global_step)
			else:
				worse_step += 1
				prefix = config.checkpoint_dir + "/" + model_name + config.tmp_model_dir
				model.tmp_saver.save(model.sess, prefix + "/tmp_{}".format(epoch), global_step=global_step)
			if global_step > config.num_training_step or worse_step > model.early_stopping:
				break
			else:
				batch = dataset.get_batch(dataset.train)
			epoch += 1
			logger.info("\nepoch {}".format(epoch))

def train(model_name, restore=True):#训练时修改Dataset中prepare_dataset
	import_lib()
	global config, logger
	config = Config.config
	dataset = Dataset.EPWDataset()
	dataset.prepare_dataset()
	logger = utils.get_logger(model_name)

	model = PHVM.PHVM(len(dataset.vocab.id2featCate), len(dataset.vocab.id2featVal), len(dataset.vocab.id2word),
					  len(dataset.vocab.id2category),
					  key_wordvec=None, val_wordvec=None, tgt_wordvec=dataset.vocab.id2vec,
					  type_vocab_size=len(dataset.vocab.id2type))
	init = {'epoch': 0, 'worse_step': 0}
	if restore:#寻找每个文件夹下最新保存的模型
		init['epoch'], init['worse_step'], model = model_utils.restore_model(model,
											config.checkpoint_dir + "/" + model_name + config.tmp_model_dir,
											config.checkpoint_dir + "/" + model_name + config.best_model_dir)
	config.check_ckpt(model_name)
	summary = tf.summary.FileWriter(config.summary_dir, model.graph)
	_train(model_name, model, dataset, summary, init)
	logger.info("finish training {}".format(model_name))

def get_args():
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda x : x.lower() == 'true')
	parser.add_argument("--cuda_visible_devices", type=str, default='0,1,2,3')
	parser.add_argument("--train", type="bool", default=True)
	parser.add_argument("--restore", type="bool", default=False)
	parser.add_argument("--model_name", type=str, default="PHVM")
	parser.add_argument("--gen_model", type=str, default="PHVM")
	parser.add_argument("--make_test", type=str, default="make_test_file.jsonl")
	#parser.add_argument("--input_topic", type="bool", default=True)
	args = parser.parse_args(sys.argv[1:])
	return args

def gen_sen(make_test, hide_strategy, STRATEGY, gen_model=None):#输入手动输入的二元组 / 生成句子模型名称，自命名区分
	"""
	:param make_test:
	:param gen_model:
	:return:
	"""
	import_lib()
	from Models.model_utils import to_testfile, load_data
	dataset = Dataset.EPWDataset()
	model = PHVM.PHVM(len(dataset.vocab.id2featCate), len(dataset.vocab.id2featVal), len(dataset.vocab.id2word),
					  len(dataset.vocab.id2category), hide_strategy=hide_strategy, STRATEGY=STRATEGY,
					  key_wordvec=None, val_wordvec=None, tgt_wordvec=dataset.vocab.id2vec,
					  type_vocab_size=len(dataset.vocab.id2type))
	config = Config.config
	# to_testfile(load_data(config.data_dir + '/topics/' + make_test))  # 制造训练集
	to_testfile(make_test)
	best_checkpoint_dir = config.checkpoint_dir + "/PHVM" + config.best_model_dir
	tmp_checkpoint_dir = config.checkpoint_dir + "/PHVM" + config.tmp_model_dir
	model_utils.restore_model(model, best_checkpoint_dir, tmp_checkpoint_dir)  # 当最优模型is not None，选择最优模型进行求解

	dataset.prepare_dataset()
	_bitfile = load_data(config.bit_file)

	_bitfile.extend(_bitfile)
	_bitfile.extend(_bitfile)
	_bitfile.extend(_bitfile)
	_bitfile.extend(_bitfile)

	bitfile = []
	if config.bit_per_word == 1 or hide_strategy == 'AC':
		bitfile = _bitfile
	elif hide_strategy == 'RS':
		for bit in range(int(len(_bitfile) / config.bit_per_word)):
			lower = bit * config.bit_per_word
			upper = lower + config.bit_per_word
			bitfile.append(bits2int(_bitfile[lower: upper]))
	texts, texts_id, bit_len = infer(model, dataset, dataset.test, bitfile)  # 输出生成文本
	# dump(texts, config.result_dir + "/{}.json".format(gen_model))
	return texts_id, bit_len

def main():
	args = get_args()

	if args.train:
		train(args.model_name, args.restore)
	else:
		gen_sen(args.make_test,'RS', args.gen_model)
		"""
		import_lib()
		from Models.model_utils import to_testfile, load_data
		dataset = Dataset.EPWDataset()
		model = PHVM.PHVM(len(dataset.vocab.id2featCate), len(dataset.vocab.id2featVal), len(dataset.vocab.id2word),
						  len(dataset.vocab.id2category),
						  key_wordvec=None, val_wordvec=None, tgt_wordvec=dataset.vocab.id2vec,
						  type_vocab_size=len(dataset.vocab.id2type))
		config = Config.config
		# to_testfile(load_data(config.make_test_file))#制造训练集
		to_testfile(load_data(Config.config.data_dir + '/topics' + args.make_test))  # 制造训练集
		best_checkpoint_dir = config.checkpoint_dir + "/" + args.model_name + config.best_model_dir
		tmp_checkpoint_dir = config.checkpoint_dir + "/" + args.model_name + config.tmp_model_dir
		model_utils.restore_model(model, best_checkpoint_dir, tmp_checkpoint_dir)  # 当最优模型is not None，选择最优模型进行求解

		dataset.prepare_dataset()
		texts = infer(model, dataset, dataset.test)  # 输出生成文本
		dump(texts, config.result_dir + "/{}.json".format(args.gen_model))
		# eval_seldel.main_eval()
		utils.print_out("finish file test")
		"""
if __name__ == "__main__":
	main()
