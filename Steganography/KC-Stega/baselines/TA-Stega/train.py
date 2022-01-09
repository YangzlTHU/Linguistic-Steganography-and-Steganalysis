from model.mta import *
from utils import *
from model.model_util import *
from config import config
import torch
from torch import nn, autograd, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import sys
from tqdm import tqdm
import os
from  dataset import Dataset
import math


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run(dataset, save_folder, load_ckpt_sign=False):
    test_topics = ['阔腿裤', '宽松', '线条']
    dataset.build_vocab()
    if torch.cuda.is_available():
        device_num = 0
        deviceName = f"cuda: {device_num}"
        torch.cuda.set_device(device_num)
        print(f'Current device: {torch.cuda.current_device()}')
    else:
        deviceName = "cpu"
    device = torch.device(deviceName)


    vocab = torch.load(f"{save_folder}/vocab.pkl")
    word_vec = torch.load(f"{save_folder}/word_vec.pkl")
    model = MTALSTM(hidden_dim=config.hidden_dim, embed_dim=config.embedding_dim, num_keywords=config.num_keywords,
					num_layers=config.num_layers, num_labels=len(vocab), weight=word_vec, vocab_size=len(vocab),
					bidirectional=config.bidirectional)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_function = nn.NLLLoss()
    if load_ckpt_sign:
        loss_values, epoch_values, bleu_values = load_ckpt_train(50, save_folder, model, device, optimizer)
    else:
        loss_values = []
        epoch_values = []
        bleu_values = []
    since = time.time()
    autograd.set_detect_anomaly(False)
    autograd.set_detect_anomaly(False)
    prev_epoch = 0 if not epoch_values else epoch_values[-1]
    best_bleu = 0 if not bleu_values else max(bleu_values)
    prev_epoch = 0 if not epoch_values else epoch_values[-1]
    best_bleu = 0 if not bleu_values else max(bleu_values)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=2, min_lr=1e-7, verbose=True)
    if config.use_gpu:
        model = model.to('cuda')
        print("Dump to cuda")
    model.apply(params_init_uniform)
    autograd.set_detect_anomaly(False)
    prev_epoch = 0 if not epoch_values else epoch_values[-1]
    best_bleu = 0 if not bleu_values else max(bleu_values)
    corpus_indice, topics_indice, corpus_test, topics_test, w2i, i2w = dataset.extract_sents(vocab)
    length = list(map(lambda x: len(x), corpus_indice))
    for epoch in range(config.num_epoch - prev_epoch):
        epoch += prev_epoch
        start = time.time()
        num, total_loss = 0, 0
        topics_indice, corpus_indice = dataset.shuffleData(topics_indice, corpus_indice) # shuffle data at every epoch
        data = dataset.data_iterator(corpus_indice, topics_indice, config.batch_size, max(length) + 1)
        hidden = model.init_hidden(batch_size=config.batch_size)
        weight = torch.ones(len(vocab))
        weight[0] = 0
        num_iter = len(corpus_indice) // config.batch_size
        for X, Y, mask, topics in tqdm(data, total=num_iter):
            num += 1
            if config.use_gpu:
                X = X.to(device)
                Y = Y.to(device)
                mask = mask.to(device)
                topics = topics.to(device)
                loss_function = loss_function.to(device)
                weight = weight.to(device)
            optimizer.zero_grad()
            coverage_vector = model.init_coverage_vector(config.batch_size, config.num_keywords)
            init_output = torch.zeros(config.batch_size, config.hidden_dim).to(device)
            output, _, hidden, _, _ = model(inputs=X, topics=topics, output=init_output, hidden=hidden, mask=mask, target=Y, coverage_vector=coverage_vector)
            hidden[0].detach_()
            hidden[1].detach_()
	        
            loss = (-output.output).reshape((-1, config.batch_size)).t() * mask
            loss = loss.sum(dim=1) / mask.sum(dim=1)
            loss = loss.mean()
            loss.backward()

            norm = 0.0
            nn.utils.clip_grad_value_(model.parameters(), 1)

            optimizer.step()
            total_loss += float(loss.item())
	        
            if np.isnan(total_loss):
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    print(name, p)
                assert False, "Gradient explode"

        one_iter_loss = np.mean(total_loss)
        lr_scheduler.step(one_iter_loss)


        # validation
        bleu_score = 0
        num_test = 500
        bleu_score = evaluate_bleu(model, topics_test, corpus_test, num_test=num_test, method='predict_rnn', i2w=i2w, w2i=w2i)

        bleu_values.append(bleu_score)
        loss_values.append(total_loss / num)
        epoch_values.append(epoch+1)

        # save checkpoint
        if ((epoch + 1) % config.check_point == 0) or (epoch == (config.num_epoch - 1)) or epoch+1 > 90 or bleu_score > 4:
            model_check_point = '%s/model_trainable_%d.pk' % (save_folder, epoch+1)
            optim_check_point = '%s/optim_trainable_%d.pkl' % (save_folder, epoch+1)
            loss_check_point = '%s/loss_trainable_%d.pkl' % (save_folder, epoch+1)
            epoch_check_point = '%s/epoch_trainable_%d.pkl' % (save_folder, epoch+1)
            bleu_check_point = '%s/bleu_trainable_%d.pkl' % (save_folder, epoch+1)
            torch.save(model.state_dict(), model_check_point)
            torch.save(optimizer.state_dict(), optim_check_point)
            torch.save(loss_values, loss_check_point)
            torch.save(epoch_values, epoch_check_point)
            torch.save(bleu_values, bleu_check_point)


        # save currunt best result
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            print('current best bleu: %.4f' % best_bleu)
            model_check_point = '%s/model_best_%d.pk' % (save_folder, epoch+1)
            optim_check_point = '%s/optim_best_%d.pkl' % (save_folder, epoch+1)
            loss_check_point = '%s/loss_best_%d.pkl' % (save_folder, epoch+1)
            epoch_check_point = '%s/epoch_best_%d.pkl' % (save_folder, epoch+1)
            bleu_check_point = '%s/bleu_best_%d.pkl' % (save_folder, epoch+1)
            torch.save(model.state_dict(), model_check_point)
            torch.save(optimizer.state_dict(), optim_check_point)
            torch.save(loss_values, loss_check_point)
            torch.save(epoch_values, epoch_check_point)
            torch.save(bleu_values, bleu_check_point)

        end = time.time()
        s = end - since
        h = math.floor(s / 3600)
        m = s - h * 3600
        m = math.floor(m / 60)
        s -= (m * 60 + h * 3600)

        if ((epoch + 1) % config.verbose == 0) or (epoch == (config.num_epoch - 1)):
            print('epoch %d/%d, loss %.4f, norm %.4f, predict bleu: %.4f, time %.3fs, since %dh %dm %ds'
                  % (epoch + 1, config.num_epoch, total_loss / num, norm, bleu_score, end - start, h, m, s))
            evaluateAndShowAttention(test_topics, model, i2w, w2i, epoch, dataset.dataname, method='beam_search')

def get_args():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda x : x.lower() == 'true')
    parser.add_argument("--dataname", type=str, default='clothes')
    parser.add_argument("--load_ckpt", type=bool, default=False)
    args = parser.parse_args(sys.argv[1:])
    return args

if __name__=="__main__":
    args = get_args()
    ckpt_root = "./ckpt"
    os.makedirs(ckpt_root, exist_ok=True)
    dataset = Dataset(args.dataname)
    save_folder = f"{ckpt_root}/{args.dataname}"
    run(dataset=dataset, save_folder=save_folder, load_ckpt_sign=args.load_ckpt)
