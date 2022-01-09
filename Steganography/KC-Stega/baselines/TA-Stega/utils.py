import numpy as np
import torch
from torch import nn, autograd, optim
import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

import codecs
import json

def load_data(input_file):
    data_list = []
    with codecs.open(input_file, 'r', encoding='UTF-8') as f:
        for line in f:
            line_data = json.loads(line.encode('utf-8'))
            data_list.append(line_data)
    return data_list

def tojson(t, path):
    file = open(path, 'w')
    for i in t:
        json_i = json.dumps(i, ensure_ascii=False)
        file.write(json_i+'\n')
    file.close()

def load_ckpt_train(version_num, save_folder, model, device, optimizer, Type = 'trainable'):
	# save_folder = 'model_result_multi_layer'
	model_check_point = '%s/model_%s_%d.pk' % (save_folder, Type, version_num)
	optim_check_point = '%s/optim_%s_%d.pkl' % (save_folder, Type, version_num)
	loss_check_point = '%s/loss_%s_%d.pkl' % (save_folder, Type, version_num)
	epoch_check_point = '%s/epoch_%s_%d.pkl' % (save_folder, Type, version_num)
	bleu_check_point = '%s/bleu_%s_%d.pkl' % (save_folder, Type, version_num)
	loss_values = []
	epoch_values = []
	bleu_values = []
	if os.path.isfile(model_check_point):
		print('Loading previous status (ver.%d)...' % version_num)
		model.load_state_dict(torch.load(model_check_point, map_location='cpu'))
		model.to(device)
		optimizer.load_state_dict(torch.load(optim_check_point))
		lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=2, min_lr=1e-7, verbose=True)
		loss_values = torch.load(loss_check_point)
		epoch_values = torch.load(epoch_check_point)
		bleu_values = torch.load(bleu_check_point)
		print('Load successfully')
	else:
		print("ver.%d doesn't exist" % version_num)
	return loss_values, epoch_values, bleu_values

def load_ckpt_eval(version_num, save_folder, model, device, Type = 'trainable'):
	# save_folder = 'model_result_multi_layer'
	model_check_point = '%s/model_%s_%d.pk' % (save_folder, Type, version_num)
	if os.path.isfile(model_check_point):
		print('Loading previous status (ver.%d)...' % version_num)
		model.load_state_dict(torch.load(model_check_point, map_location='cpu'))
		model.to(device)
		print('Load successfully')
	else:
		print("ver.%d doesn't exist" % version_num)

def evaluate_bleu(model, topics_test, corpus_test, num_test, i2w, w2i, method='beam_search'):
    bleu_2_score = 0
    for i in tqdm(range(len(corpus_test[:num_test]))):
        if method == 'beam_search':
            _, output_words, _, _ = model.beam_search([i2w[x] for x in topics_test[i]], 100, model, i2w, w2i, False)
        else:
            _, output_words, _, _ = model.predict_rnn([i2w[x] for x in topics_test[i]], 100, model, i2w, w2i,)
        bleu_2_score += sentence_bleu([[i2w[x] for x in corpus_test[i] if x not in [0, 2]]], output_words, weights=(0, 1, 0, 0))
        
    bleu_2_score = bleu_2_score / num_test * 100
    return bleu_2_score


def showAttention(input_sentence, output_words, attentions, ite, dataset):
    att_vis_dir = f'./att_vis/{dataset}'
    os.makedirs(att_vis_dir)
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.subplots(1)
    #     cmap = 'bone'
    cmap = 'viridis'
    cax = ax.matshow(attentions.numpy(), cmap=cmap)
    fig.colorbar(cax)

    # Set up axes
    ax.set_yticklabels([''] + input_sentence.split(' '), fontsize=10)
    ax.set_xticklabels([''] + output_words, fontsize=10, rotation=45)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    word_size = 0.5
    fig.set_figheight(word_size * len(input_sentence.split(' ')))
    fig.set_figwidth(word_size * len(output_words))
    plt.savefig(f"{att_vis_dir}/att_vis{ite}.jpg")


def evaluateAndShowAttention(input_sentence, model, i2w, w2i, iter, dataset_name, method='beam_search', is_sample=False):
    num_char = 100
    if method == 'beam_search':
        _, output_words, attentions, coverage_vector = model.beam_search(input_sentence, num_char, model, i2w, w2i,
                                                                         is_sample=is_sample)
    else:
        _, output_words, attentions, _ = model.predict_rnn(input_sentence, num_char, model, i2w, w2i)
    print('input =', ' '.join(input_sentence))
    print('output =', ' '.join(output_words))
    #     n_digits = 3
    #     coverage_vector = torch.round(coverage_vector * 10**n_digits) / (10**n_digits)
    #     coverage_vector=np.round(coverage_vector, n_digits)
    #     print(coverage_vector.numpy())
    showAttention(' '.join(input_sentence), output_words, attentions, iter, dataset_name)

def params_init_uniform(m):
    if type(m) == nn.Linear:
        y = 0.04
        nn.init.uniform_(m.weight, -y, y)


def decay_lr(optimizer, epoch, factor=0.1, lr_decay_epoch=60):
    if epoch % lr_decay_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * factor
        print('lr decayed to %.4f' % optimizer.param_group[0]['lr'])
    return optimizer

def get_bitstream():
    with open('./bit_stream/bit_stream.txt', 'r', encoding='utf8') as f:
        bit_stream = f.read().strip()
        bit_stream += bit_stream
        bit_stream += bit_stream
        bit_stream += bit_stream
        bit_stream += bit_stream
    return bit_stream

# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit*(2**i)
    return res


def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i
