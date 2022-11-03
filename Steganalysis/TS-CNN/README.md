# Abstract
Steganalysis has been an important research topic in cyberse-
curity that helps to identify covert attacks in public network.
With the rapid development of natural language processing
technology in the past two years, coverless steganography has
been greatly developed. Previous text steganalysis methods
have shown unsatisfactory results on this new steganography
technique and remain an unsolved challenge. Different from
all previous text steganalysis methods, in this paper, we propose a text steganalysis method(TS-CNN) based on semantic analysis, which uses convolutional neural network(CNN)
to extract high-level semantic features of texts, and ﬁnds the
subtle distribution differences in the semantic space before
and after embedding the secret information. To train and test
the proposed model, we collected and released a large text
steganalysis(CT-Steg) dataset, which contains a total number
of 216,000 texts with various lengths and various embedding
rates. Experimental results show that the proposed model can
achieve nearly 100% precision and recall, outperforms all the
previous methods. Furthermore, the proposed model can even
estimate the capacity of the hidden information inside. These
results strongly support that using the subtle changes in the
semantic space before and after embedding the secret information to conduct text steganalysis is feasible and effective.


# Method
<div align=center><img src=https://github.com/YangzlTHU/Linguistic-Steganography-and-Steganalysis/blob/master/Steganalysis/pics/ts-csw.jpg  width="400" height="300"></div>

# Experiments
<div align=center><img src=https://github.com/YangzlTHU/Linguistic-Steganography-and-Steganalysis/blob/master/Steganalysis/TS-CNN/table3.png  width="600" height="400"></div>


# Codes
Codes can be found in the collections [TextSteganalysis](https://github.com/yjs1224/TextSteganalysis)

## For details of the methods and results, please refer to our paper.

```bibtex 
@article{yang2020ts,
  title={TS-CSW: text steganalysis and hidden capacity estimation based on convolutional sliding windows},
  author={Yang, Zhongliang and Huang, Yongfeng and Zhang, Yu-Jin},
  journal={Multimedia Tools and Applications},
  volume={79},
  number={25},
  pages={18293--18316},
  year={2020},
  publisher={Springer}
}
```
