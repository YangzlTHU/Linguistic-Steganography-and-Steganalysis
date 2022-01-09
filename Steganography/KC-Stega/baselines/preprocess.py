import gensim
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
import torch
from random import shuffle
import jieba
import logging
import pandas as pd
from gensim.models import word2vec
import argparse
import sys
import os

def write_seg_sentence(file_path, dataname):
	output = open(f'{file_path}/{dataname}_seg.txt', 'w', encoding='utf-8')
	num_lines = sum(1 for line in open(f'{file_path}/{dataname}.txt', 'r'))
	with open(f'{file_path}/{dataname}.txt') as f:
		for idx, line in tqdm(enumerate(f), total=num_lines):
			if idx > 305000:
				print('\nextract %d articles' % idx)
				break
			article = line.strip('\n')
			article, topics = article.split(' </d> ')
			output.write(article)
			output.write(' \n')
		f.close()
	    
	output.close()

def w2v(file_path, dataname):
	word2vec_params = {
		'sg': 1,
		"size": 100,
		"alpha": 0.01,
		"min_alpha": 0.0005,
		'window': 10,
		'min_count': 1,
		'seed': 1,
		"workers": 24,
		"negative": 0,
		"hs": 1,
		'compute_loss': True,
		'iter': 50,
		'cbow_mean': 0,
	}
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	sentences = word2vec.LineSentence(f'{file_path}/{dataname}_seg.txt')
	model = word2vec.Word2Vec(sentences=sentences, **word2vec_params)
	model.save(f'{file_path}/{dataname}_mincount_1_305000_vec_original.model')
	out = f'{file_path}/{dataname}_mincount_1_305000_vec_original.txt'
	model.wv.save_word2vec_format(out, binary=False)

def bulid_vocab(file_path, dataname):
	fvec = KeyedVectors.load_word2vec_format(f'{file_path}/{dataname}_mincount_1_305000_vec_original.txt', binary=False)
	word_vec = fvec.vectors
	vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
	vocab.extend(list(fvec.vocab.keys()))
	word_vec = np.concatenate((np.array([[0]*word_vec.shape[1]] * 4), word_vec))
	word_vec = torch.tensor(word_vec).float()
	print("total %d words" % len(word_vec))
	word_to_idx = {ch: i for i, ch in enumerate(vocab)}
	idx_to_word = {i: ch for i, ch in enumerate(vocab)}
	return vocab, word_vec, word_to_idx, idx_to_word

def save_vec(save_folder, vocab, word_vec):
	vocab_check_point = '%s/vocab.pkl' % save_folder
	word_vec_check_point = '%s/word_vec.pkl' % save_folder
	torch.save(vocab, vocab_check_point)
	torch.save(word_vec, word_vec_check_point)

def get_args():
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda x : x.lower() == 'true')
	parser.add_argument("--file_path", type=str, default='./data')
	parser.add_argument("--dataname", type=str, default='clothes')
	args = parser.parse_args(sys.argv[1:])
	return args


if __name__ == "__main__":
	args = get_args()
	save_folder = f'./ckpt/{args.dataname}'
	os.makedirs(save_folder, exist_ok=True)
	file_path = f"{args.file_path}/{args.dataname}"
	write_seg_sentence(file_path, args.dataname)
	w2v(file_path, args.dataname)
	vocab, word_vec, _, _ = bulid_vocab(file_path, args.dataname)
	save_vec(save_floder, vocab, word_vec)
	print("finish building vocab vectors. ")

