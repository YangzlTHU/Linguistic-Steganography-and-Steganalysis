# SeSy: Linguistic Steganalysis Framework Integrating Semantic and Syntactic Features

# Abstract
With the rapid development of natural language
processing technology and linguistic steganography, linguistic
steganalysis gains considerable interest in recent years. Current
advanced methods dominantly focus on statistical features in
semantic view yet ignore syntax structure of text, which leads to
limited performance to some newly statistically indistinguishable
steganography algorithms. To ﬁll this gap, in this paper, we
propose a novel linguistic steganalysis framework named SeSy to
integrate both semantic and syntactic features. Speciﬁcally, we
propose to employ transformer-architecture language model as
semantics extractor and leverage a graph attention network to
retain syntactic features. Extensive experimental results show that
owing to additional syntactic information, the SeSy framework
effectively brings about remarkable improvement to current
advanced linguistic steganalysis methods.

# Motivation
<div align=center><img src=https://github.com/YangzlTHU/Linguistic-Steganography-and-Steganalysis/blob/master/Steganalysis/pics/sesy.jpg  width="500" height="300"></div>

  Despite excellent improvement, existing methods only focus
on statistical features from the semantic aspect. In fact, varied
aspects of features seriously deviate from features of normal
texts when embedding information. We notice that the syntactic structure of text can also be changed when embedding
information. When generating words,
different secret information may lead to different syntactic
structures. Such a fact implies that syntactic structure is
signiﬁcant for linguistic steganalysis. However, almost all the
existing methods only model text sequentially yet neglect
syntactic information within text carrier, which limits detection
performance seriously. Subject to serial modelling, these methods can only regard text carrier as sequential tokens in a ﬁxed
order, such as left-to-right and right-to-left order. Though Wu
et al.  innovatively use GNN to model text, features that
they analyse are still conﬁned to statistical relevance among
sequential words. Therefore, when it comes to some newly
statistically indistinguishable steganography algorithms,
they encounter limited performance especially. Besides, serial
modelling in recent NN-based works will also
damage the capacity of discovering statistical discrepancy in
semantic aspect, since the limited scope weakens the capacity
of holding complex long-distance semantic representation.

# Framework
<div align=center><img src=https://github.com/YangzlTHU/Linguistic-Steganography-and-Steganalysis/blob/master/Steganalysis/pics/sesy2.jpg  width="400" height="300"></div>
The overall architecture of the proposed SeSy framework. Semantics
extracting component extracts complex semantics from text input, which is
subsequently be used to initial vertexes of graph. Edges of graph are initialized
by structure of dependency parsing tree. Then syntactic representation will be
captured by syntax retaining component. Finally adopting different strategies
named Cas and Parl makes us acquire integrated features, on which dis-
criminator makes a decision whether the carrier under test is steganographic.

# Experiments

<div align=center><img src=https://github.com/YangzlTHU/Linguistic-Steganography-and-Steganalysis/blob/master/Steganalysis/SeSy/sesy-table1.png  width="700" height="300"></div>

# Codes
[SeSy](https://github.com/yjs1224/SeSy) or [TextSteganalysis](https://github.com/yjs1224/TextSteganalysis)

## For details of the methods and results, please refer to our paper.

```bibtex 
@article{yang2021sesy,
  title={SeSy: Linguistic Steganalysis Framework Integrating Semantic and Syntactic Features},
  author={Yang, Jinshuai and Yang, Zhongliang and Zhang, Siyu and Tu, Haoqin and Huang, Yongfeng},
  journal={IEEE Signal Processing Letters},
  volume={29},
  pages={31--35},
  year={2021},
  publisher={IEEE}
}
```
