<p align="center">
  <img src="docs/fairseq_logo.png" width="150">
  <br />
  <br />
  <a href="https://github.com/pytorch/fairseq/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/pytorch/fairseq.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/actions?query=workflow:build"><img alt="Build Status" src="https://github.com/pytorch/fairseq/workflows/build/badge.svg" /></a>
  <a href="https://fairseq.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/fairseq/badge/?version=latest" /></a>
</p>

--------------------------------------------------------------------------------

Fairseq(-py)是一个序列建模工具包，研究人员和开发人员可以使用它们来训练定制模型，以进行翻译，摘要，语言建模和其他文本生成任务。


我们提供各种序列建模文件的参考实现：

<details><summary>一些论文</summary><p>

* **卷积神经网络 (CNN)**
  + [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](examples/language_model/conv_lm/README.md)
  + [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](examples/conv_seq2seq/README.md)
  + [Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018)](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
  + [Hierarchical Neural Story Generation (Fan et al., 2018)](examples/stories/README.md)
  + [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
* **LightConv and DynamicConv（动态卷积） 模型， **
  + [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](examples/pay_less_attention_paper/README.md)
* **Long Short-Term Memory (LSTM) networks**
  + Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)
* **Transformer (self-attention) networks**
  + Attention Is All You Need (Vaswani et al., 2017)
  + [Scaling Neural Machine Translation (Ott et al., 2018)](examples/scaling_nmt/README.md)
  + [Understanding Back-Translation at Scale (Edunov et al., 2018)](examples/backtranslation/README.md)
  + [Adaptive Input Representations for Neural Language Modeling (Baevski and Auli, 2018)](examples/language_model/README.adaptive_inputs.md)
  + [Lexically constrained decoding with dynamic beam allocation (Post & Vilar, 2018)](examples/constrained_decoding/README.md)
  + [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](examples/translation_moe/README.md)
  + [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](examples/roberta/README.md)
  + [Facebook FAIR's WMT19 News Translation Task Submission (Ng et al., 2019)](examples/wmt19/README.md)
  + [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](examples/joint_alignment_translation/README.md )
  + [Multilingual Denoising Pre-training for Neural Machine Translation (Liu et at., 2020)](examples/mbart/README.md)
  + [Neural Machine Translation with Byte-Level Subwords (Wang et al., 2020)](examples/byte_level_bpe/README.md)
  + [Unsupervised Quality Estimation for Neural Machine Translation (Fomicheva et al., 2020)](examples/unsupervised_quality_estimation/README.md)
  + [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](examples/wav2vec/README.md)
  + [Generating Medical Reports from Patient-Doctor Conversations Using Sequence-to-Sequence Models (Enarvi et al., 2020)](examples/pointer_generator/README.md)
  + [Linformer: Self-Attention with Linear Complexity (Wang et al., 2020)](examples/linformer/README.md)
  + [Cross-lingual Retrieval for Iterative Self-Supervised Training (Tran et al., 2020)](examples/criss/README.md)
  + [Deep Transformers with Latent Depth (Li et al., 2020)](examples/latent_depth/README.md)
* **Non-autoregressive（非自回归） Transformers**
  + Non-Autoregressive Neural Machine Translation (Gu et al., 2017)
  + Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement (Lee et al. 2018)
  + Insertion Transformer: Flexible Sequence Generation via Insertion Operations (Stern et al. 2019)
  + Mask-Predict: Parallel Decoding of Conditional Masked Language Models (Ghazvininejad et al., 2019)
  + [Levenshtein Transformer (Gu et al., 2019)](examples/nonautoregressive_translation/README.md)
* **Finetuning**
  + [Better Fine-Tuning by Reducing Representational Collapse (Aghajanyan et al. 2020)](examples/rxf/README.md)

</p></details>

### What's New:

* November 2020: Adopted [Hydra](https://github.com/facebookresearch/hydra) as a configuration framework;
[added documentation explaining how to use it for new and existing projects](docs/hydra_integration.md)
* October 2020: [Added R3F/R4F (Better Fine-Tuning) code](examples/rxf/README.md)
* October 2020: [Deep Transformer with Latent Depth code released](examples/latent_depth/README.md)
* October 2020: [Added CRISS models and code](examples/criss/README.md)
* September 2020: [Added Linformer code](examples/linformer/README.md)
* September 2020: [Added pointer-generator networks](examples/pointer_generator/README.md)
* August 2020: [Added lexically constrained decoding](examples/constrained_decoding/README.md)
* August 2020: [wav2vec2 models and code released](examples/wav2vec/README.md)
* July 2020: [Unsupervised Quality Estimation code released](examples/unsupervised_quality_estimation/README.md)
* May 2020: [Follow fairseq on Twitter](https://twitter.com/fairseq)
* April 2020: [Monotonic Multihead Attention code released](examples/simultaneous_translation/README.md)
* April 2020: [Quant-Noise code released](examples/quant_noise/README.md)
* April 2020: [Initial model parallel support and 11B parameters unidirectional LM released](examples/megatron_11b/README.md)

<details><summary>Previous updates</summary><p>

* March 2020: [Byte-level BPE code released](examples/byte_level_bpe/README.md)
* February 2020: [mBART model and code released](examples/mbart/README.md)
* February 2020: [Added tutorial for back-translation](https://github.com/pytorch/fairseq/tree/master/examples/backtranslation#training-your-own-model-wmt18-english-german)
* December 2019: [fairseq 0.9.0 released](https://github.com/pytorch/fairseq/releases/tag/v0.9.0)
* November 2019: [VizSeq released (a visual analysis toolkit for evaluating fairseq models)](https://facebookresearch.github.io/vizseq/docs/getting_started/fairseq_example)
* November 2019: [CamemBERT model and code released](examples/camembert/README.md)
* November 2019: [BART model and code released](examples/bart/README.md)
* November 2019: [XLM-R models and code released](examples/xlmr/README.md)
* September 2019: [Nonautoregressive translation code released](examples/nonautoregressive_translation/README.md)
* August 2019: [WMT'19 models released](examples/wmt19/README.md)
* July 2019: fairseq relicensed under MIT license
* July 2019: [RoBERTa models and code released](examples/roberta/README.md)
* June 2019: [wav2vec models and code released](examples/wav2vec/README.md)

</p></details>

### 特点:

* 一台机器或多台机器上的多GPU训练(数据和模型并行)
* 通过实现多种搜索算法在CPU和GPU上快速生成：
  + beam search
  + Diverse Beam Search ([Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424))
  + sampling (unconstrained, top-k and top-p/nucleus)
  + lexically constrained decoding ([Post & Vilar, 2018](examples/constrained_decoding/README.md))
* 通过延迟更新，即使在单个GPU上，也可以进行大型的mini-batch训练
* 混合精度训练  (在较少的GPU内存的情况下训练更快 [NVIDIA tensor cores](https://developer.nvidia.com/tensor-cores))
* 可扩展：轻松注册新模型，标准，任务，优化器和学习率调度程序
* [flexible configuration](docs/hydra_integration.md) based on [Hydra](https://github.com/facebookresearch/hydra) allowing a combination of code, command-line and file based configuration

We also provide [pre-trained models for translation and language modeling](#pre-trained-models-and-examples)
带有方便的`torch.hub`界面：

``` python
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
en2de.translate('Hello world', beam=5)
# 'Hallo Welt'
```

请参阅PyTorch Hub教程以了解  [translation](https://pytorch.org/hub/pytorch_fairseq_translation/)
and [RoBERTa](https://pytorch.org/hub/pytorch_fairseq_roberta/) for more examples.

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **安装 fairseq** 和本地开发:

``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# MacOS上的安装方法:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* 如果您使用Docker，请确保通过以下任一方法增加共享内存大小  `--ipc=host` or `--shm-size`
作为命令行选项 `nvidia-docker run` .

# 快速开始

The [full documentation](https://fairseq.readthedocs.io/) 包含有关入门，训练新模型以及使用新模型类型和任务扩展fairseq的说明。

# 预训练模型和示例

我们为以下列出的几个任务提供了经过预训练的模型和经过预处理的二分类测试集，以及样本的训练和评估命令。

* [Translation](examples/translation/README.md): 卷积模型和transformer模型
* [Language Modeling](examples/language_model/README.md): 卷积模型和transformer模型

我们还提供了更详细的READMEs，以重现特定论文的结果：

* [Cross-lingual Retrieval for Iterative Self-Supervised Training (Tran et al., 2020)](examples/criss/README.md)
* [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](examples/wav2vec/README.md)
* [Unsupervised Quality Estimation for Neural Machine Translation (Fomicheva et al., 2020)](examples/unsupervised_quality_estimation/README.md)
* [Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)](examples/quant_noise/README.md)
* [Neural Machine Translation with Byte-Level Subwords (Wang et al., 2020)](examples/byte_level_bpe/README.md)
* [Multilingual Denoising Pre-training for Neural Machine Translation (Liu et at., 2020)](examples/mbart/README.md)
* [Reducing Transformer Depth on Demand with Structured Dropout (Fan et al., 2019)](examples/layerdrop/README.md)
* [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](examples/joint_alignment_translation/README.md)
* [Levenshtein Transformer (Gu et al., 2019)](examples/nonautoregressive_translation/README.md)
* [Facebook FAIR's WMT19 News Translation Task Submission (Ng et al., 2019)](examples/wmt19/README.md)
* [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](examples/roberta/README.md)
* [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
* [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](examples/translation_moe/README.md)
* [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](examples/pay_less_attention_paper/README.md)
* [Understanding Back-Translation at Scale (Edunov et al., 2018)](examples/backtranslation/README.md)
* [Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018)](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
* [Hierarchical Neural Story Generation (Fan et al., 2018)](examples/stories/README.md)
* [Scaling Neural Machine Translation (Ott et al., 2018)](examples/scaling_nmt/README.md)
* [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](examples/conv_seq2seq/README.md)
* [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](examples/language_model/README.conv.md)

# Join the fairseq community

* Twitter: https://twitter.com/fairseq
* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

# License

fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
