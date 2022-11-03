# Abstract
With the rapid development of natural language
processing technologies, more and more text steganographic
methods based on automatic text generation technology have
appeared in recent years. These models use the powerful self-
learning and feature extraction ability of the neural networks
to learn the feature expression of massive normal texts. Then
they can automatically generate dense steganographic texts which
conform to such statistical distribution based on the learned
statistical patterns. In this paper, we observe that the conditional probability distribution of each word in the automatically
generated steganographic texts will be distorted after embedded
with hidden information. We use Recurrent Neural Networks
(RNNs) to extract these feature distribution differences and then
classify those features into cover text and stego text categories.
Experimental results show that the proposed model can achieve
high detection accuracy. Besides, the proposed model can even
make use of the subtle differences of the feature distribution of
texts to estimate the amount of hidden information embedded in
the generated steganographic text.


# Method
<div align=center><img src=https://github.com/YangzlTHU/Linguistic-Steganography-and-Steganalysis/blob/master/Steganalysis/pics/ts-rnn.jpg  width="500" height="300"></div>

# Experiments
<div align=center><img src=https://github.com/YangzlTHU/Linguistic-Steganography-and-Steganalysis/blob/master/Steganalysis/TS-RNN/table1.png  width="800" height="400"></div>

<div align=center><img src=https://github.com/YangzlTHU/Linguistic-Steganography-and-Steganalysis/blob/master/Steganalysis/TS-RNN/fig2.png  width="600" height="200"></div>

# Codes
Codes can be found in the collections [TextSteganalysis](https://github.com/yjs1224/TextSteganalysis)

## For details of the methods and results, please refer to our paper.

```bibtex 
@article{yang2019ts,
  title={TS-RNN: text steganalysis based on recurrent neural networks},
  author={Yang, Zhongliang and Wang, Ke and Li, Jian and Huang, Yongfeng and Zhang, Yu-Jin},
  journal={IEEE Signal Processing Letters},
  volume={26},
  number={12},
  pages={1743--1747},
  year={2019},
  publisher={IEEE}
}
```

