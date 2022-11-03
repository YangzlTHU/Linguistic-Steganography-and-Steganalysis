# Provably Secure Generative Linguistic Steganography
## Abstract
Generative linguistic steganography mainly utilized language models and applied steganographic sampling (stegosampling) to generate high-security steganographic text (stego-text). However, previous methods generallylead to statistical differences between the conditional probability distributions of stegotext and natural text, which brings about security risks. In this paper, to further ensure security, we present a novel provably secure generative linguistic steganographic method ADG, which recursively embeds secret information by Adaptive Dynamic Grouping of tokens according to their probability given by an off-the-shelf language model. We not only prove the security of ADG mathematically, but also conduct extensive experiments on three public corpora to further verify its imperceptibility. The experimental results reveal that the proposed method is able to generate stegotext with nearly perfect security.

## Adaptive Dynamic Grouping (ADG) 
<div align=center><img src=https://github.com/YangzlTHU/Linguistic-Steganography-and-Steganalysis/blob/master/Steganography/ADG-steganography/adg.png width="240" height="400"> </div>

## Experiments
<div align=center> <img src=https://github.com/YangzlTHU/Linguistic-Steganography-and-Steganalysis/blob/master/Steganography/ADG-steganography/table1.png  width="500" height="300"> </div>

<div align=center><img src=https://github.com/YangzlTHU/Linguistic-Steganography-and-Steganalysis/blob/master/Steganography/ADG-steganography/table2.png  width="500" height="300"></div>

<div align=center><img src=https://github.com/YangzlTHU/Linguistic-Steganography-and-Steganalysis/blob/master/Steganography/ADG-steganography/table3.png  width="500" height="300">
</div>

## codes link
[ADG-steganography](https://github.com/Mhzzzzz/ADG-steganography)

## For details of the methods and results, please refer to our paper.

```bibtex 
@inproceedings{zhang2021provably,
  title={Provably Secure Generative Linguistic Steganography},
  author={Zhang, Siyu and Yang, Zhongliang and Yang, Jinshuai and Huang, Yongfeng}, 
  booktitle={Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  pages={3046--3055}, 
  year={2021}
 }
```
