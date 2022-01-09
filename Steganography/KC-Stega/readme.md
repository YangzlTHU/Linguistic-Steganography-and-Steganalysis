## KC-Stega: Keyword-Constrained Linguistic Generative Steganography

### Prerequsites

```python
python==3.6.0
tensorflow-gpu==1.4.0
tensorflow-tensorboard==0.4.0
```

see `requirements.txt` for environment details，use command `pip install -r requirements.txt` to install all dependencies.

### Steganography and Parameter Setting

Use Taobao E-commerse dataset

1. Download `data.zip` from [Baidu Drive](https://pan.baidu.com/s/1IvftE4iP9R6gplZpLxDoMw ) with password `7k12` or from [Google Drive](https://drive.google.com/file/d/1f4mk6z7zaAeC4fxmruBeku5IZtkxCfxt/view?usp=sharing), and unzip it under the root folder.
2. Download `checkpoints.zip` from [Baidu Drive](https://pan.baidu.com/s/1dvSsvWYaD3WgVrtoUPBklg) with password `98x1` or from [Google Drive](https://drive.google.com/drive/folders/1__jYKLz4uY9A87HLhyE6DUo5yYYGNNyb?usp=sharing), and unzip it under the  `./result` folder.
3. Use the following command to generate keyword constrained stego texts.

```python
# Steganography
python eval_seldel.py \
--input_topic 3topic_norep_3k_1205_1 \ # input keyword files，under `./data/processed/topics`, only add file name without `.jsonl`
--hide_strategy AC \ # hiding strategy, AC for arithmetic coding，RS for rejection sampling coding，HC for Huffman coding (to be implemented)
--STRATEGY sample \ # candidate pool selecting strategy, sample for categorical sampling，topk for top-k sampling
--gen_num 15 \ # Max generation looping time (>3)
--gen_thresh 0.95 # keyword extraction threshhold, default as 0.95
```

All hyper-parameters can be set and checked in `Config.py `.

```python
self.bit_per_word = 2 # Bit per word
self.AC_precision = 26 # Precision of arithmetic coding
self.truncated_vocab_size=5000 # Dynamic truncated vocabunary size
```

### Output

1. log files are in `./log` folder.
2. The final stego texts are in `./result` folder.

### Baselines

Baseline model (TA-Stega) please refer to `./baselines` folder.