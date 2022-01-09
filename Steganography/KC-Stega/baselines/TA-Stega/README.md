# TA-Stega

Pytorch implementation of [**Linguistic Generative Steganography With Enhanced Cognitive-Imperceptibility**](https://ieeexplore.ieee.org/document/9353234) based on MTA-LSTM, Arithmetic Coding and KeyWord Generation (KWG) trick.

## Dataset

- Taobao E-commerce Dataset (Chinese)
- Composition and Zhihu (Chinese)

Download from [Baidu Drive](https://pan.baidu.com/s/167hCdgnE50ZpOTGJZAKQhA) with password `i597` or [Google Drive](https://drive.google.com/file/d/1EvmwmfI5y7LwpBItpjlRIp20z6xheu9E/view?usp=sharing)

## Prerequisites

- Python3
- PyTorch >= 1.2.0
- Gensim

## Usage

### To generate stego text directly

1. Download model checkpoint and vocabulary from [Google Drive](https://drive.google.com/file/d/18bdUBN5ldcr22QgFEFhLFcu_Xf47YsoG/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1IANV_thvSjOROUjGYTFhLQ) with password `jjiy`, put `clothes.zip` under `./ckpt` folder and unzip it.

2. ```bash
   python AC_stega.py \
   --dataname clothes \ # clothes for taobao dataset
   --method AC \ # steganoraphy method (AC for arithmetic is implemented)
   --Stra sample \ # candidate pool construction strategy ('topk', 'sample' or '')
   --max_bit 5 \ # max candidate pool size
   --sent_num 10 \ # generate sent_num stego sentences for each CPS config
   --kww True \ # whether use KeyWord Wise Generation trick
   --stega True # true for steganography, false for extraction
   ```

### To train the model on your own datasets

1. Sort your dataset as below:

   ```bash
   TEXT <\d> keyword1 keyword2 keyword3
   ```

   Details refer to the given training corpus.  Put your corpus file to `./data/{dataname}` folder

2. Generate pre-tained word embedding files

   ```bash
   python preprocess.py \
   --file_path ./data \ # path of corpus
   --dataname clothes
   ```

3. Train the model

   ```bash
   python train.py \
   --dataname clothes \
   --load_ckpt False # whether to load checkpoint
   ```

Training parameters please refer to `config.py`

The code adopts some practice from [this repo](https://github.com/jwang0306/mta-lstm-pytorch), Thanks a lot !!