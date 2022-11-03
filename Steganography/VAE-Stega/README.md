# Abstract
In recent years, linguistic steganography based on text auto-generation technology has been greatly developed, which is considered to be a very promising but also a very challenging research topic. Previous works mainly focus on optimizing the language model and conditional probability coding methods, aiming at generating steganographic sentences with better quality. In this paper, we ﬁrst report some of our latest experimental ﬁndings, which seem to indicate that the quality of the generated steganographic text cannot fully guarantee its steganographic security, and even has a prominent perceptual-imperceptibility and statistical-imperceptibility conﬂict effect (Psic Effect). To further improve the imperceptibility and security of generated steganographic texts, in this paper, we propose a new linguistic steganography based on Variational Auto-Encoder (VAE), which can be called VAE-Stega. We use the encoder in VAE-Stega to learn the overall statistical distribution characteristics of a large number of normal texts, and then use the decoder in VAE-Stega to generate steganographic sentences which conform to both of the statistical language model as well as the overall statistical distribution of normal sentences, so as to guarantee both the perceptual-imperceptibility and statistical-imperceptibility of the generated steganographic texts at the same time. We design several experiments to test the proposed method. Experimental results show that the proposed model can greatly improve the imperceptibility of the generated steganographic sentences and thus achieves the state of the art performance.

# Method

# Experiments

# Codes
[VAE-Stega](https://github.com/YangzlTHU/VAE-Stega)

# For details of the methods and results, please refer to our paper.
```bibtex
@article{yang2020vae,
  title={VAE-Stega: linguistic steganography based on variational auto-encoder},
  author={Yang, Zhong-Liang and Zhang, Si-Yu and Hu, Yu-Ting and Hu, Zhi-Wen and Huang, Yong-Feng},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={16},
  pages={880--895},
  year={2020},
  publisher={IEEE}
}
```
