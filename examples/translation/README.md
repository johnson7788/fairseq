# Neural Machine Translation

This README contains instructions for [using pretrained translation models](#example-usage-torchhub)
as well as [training new models](#training-a-new-model).

## Pre-trained models

模型 | 描述 | 数据 | 下载
---|---|---|---
`conv.wmt14.en-fr` | Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | model: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2) <br> newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2) <br> newstest2012/2013: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.ntst1213.tar.bz2)
`conv.wmt14.en-de` | Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) | model: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-de.fconv-py.tar.bz2) <br> newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-de.newstest2014.tar.bz2)
`conv.wmt17.en-de` | Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT17 English-German](http://statmt.org/wmt17/translation-task.html#Download) | model: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt17.v2.en-de.fconv-py.tar.bz2) <br> newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt17.v2.en-de.newstest2014.tar.bz2)
`transformer.wmt14.en-fr` | Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | model: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2) <br> newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)
`transformer.wmt16.en-de` | Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | model: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2) <br> newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)
`transformer.wmt18.en-de` | Transformer <br> ([Edunov et al., 2018](https://arxiv.org/abs/1808.09381)) <br> WMT'18 winner | [WMT'18 English-German](http://www.statmt.org/wmt18/translation-task.html) | model: <br> [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz) <br> See NOTE in the archive
`transformer.wmt19.en-de` | Transformer <br> ([Ng et al., 2019](https://arxiv.org/abs/1907.06616)) <br> WMT'19 winner | [WMT'19 English-German](http://www.statmt.org/wmt19/translation-task.html) | model: <br> [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz)
`transformer.wmt19.de-en` | Transformer <br> ([Ng et al., 2019](https://arxiv.org/abs/1907.06616)) <br> WMT'19 winner | [WMT'19 German-English](http://www.statmt.org/wmt19/translation-task.html) | model: <br> [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz)
`transformer.wmt19.en-ru` | Transformer <br> ([Ng et al., 2019](https://arxiv.org/abs/1907.06616)) <br> WMT'19 winner | [WMT'19 English-Russian](http://www.statmt.org/wmt19/translation-task.html) | model: <br> [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz)
`transformer.wmt19.ru-en` | Transformer <br> ([Ng et al., 2019](https://arxiv.org/abs/1907.06616)) <br> WMT'19 winner | [WMT'19 Russian-English](http://www.statmt.org/wmt19/translation-task.html) | model: <br> [download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz)

## Example usage (torch.hub)

我们需要一些其他Python依赖项来进行预处理:
```bash
pip install fastBPE sacremoses subword_nmt
```

通过PyTorch Hub进行交互式翻译:
```python
import torch

# 列出可用的模型
torch.hub.list('pytorch/fairseq')  

# 例如：
['bart.base',
 'bart.large',
 'bart.large.cnn',
 'bart.large.mnli',
 'bart.large.xsum',
 'bpe',
 'camembert',
 'camembert-base',
 'camembert-base-ccnet',
 'camembert-base-ccnet-4gb',
 'camembert-base-oscar-4gb',
 'camembert-base-wikipedia-4gb',
 'camembert-large',
 'camembert.v0',
 'conv.stories',
 'conv.stories.pretrained',
 'conv.wmt14.en-de',
 'conv.wmt14.en-fr',
 'conv.wmt17.en-de',
 'data.stories',
 'dynamicconv.glu.wmt14.en-fr',
 'dynamicconv.glu.wmt16.en-de',
 'dynamicconv.glu.wmt17.en-de',
 'dynamicconv.glu.wmt17.zh-en',
 'dynamicconv.no_glu.iwslt14.de-en',
 'dynamicconv.no_glu.wmt16.en-de',
 'lightconv.glu.wmt14.en-fr',
 'lightconv.glu.wmt16.en-de',
 'lightconv.glu.wmt17.en-de',
 'lightconv.glu.wmt17.zh-en',
 'lightconv.no_glu.iwslt14.de-en',
 'lightconv.no_glu.wmt16.en-de',
 'roberta.base',
 'roberta.large',
 'roberta.large.mnli',
 'roberta.large.wsc',
 'tokenizer',
 'transformer.wmt14.en-fr',
 'transformer.wmt16.en-de',
 'transformer.wmt18.en-de',
 'transformer.wmt19.de-en',
 'transformer.wmt19.de-en.single_model',
 'transformer.wmt19.en-de',
 'transformer.wmt19.en-de.single_model',
 'transformer.wmt19.en-ru',
 'transformer.wmt19.en-ru.single_model',
 'transformer.wmt19.ru-en',
 'transformer.wmt19.ru-en.single_model',
 'transformer_lm.gbw.adaptive_huge',
 'transformer_lm.wiki103.adaptive',
 'transformer_lm.wmt19.de',
 'transformer_lm.wmt19.en',
 'transformer_lm.wmt19.ru',
 'xlmr.base',
 'xlmr.large']


# Load a transformer trained on WMT'16 En-De
# Note: WMT'19 models use fastBPE instead of subword_nmt, see instructions below
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de',
                       tokenizer='moses', bpe='subword_nmt')
en2de.eval()  # disable dropout

# The underlying model is available under the *models* attribute
assert isinstance(en2de.models[0], fairseq.models.transformer.TransformerModel)

# Move model to GPU for faster translation
en2de.cuda()

# Translate a sentence
en2de.translate('Hello world!')
# 'Hallo Welt!'

# Batched translation
en2de.translate(['Hello world!', 'The cat sat on the mat.'])
# ['Hallo Welt!', 'Die Katze saß auf der Matte.']
```

加载自定义模型:
```python
from fairseq.models.transformer import TransformerModel
zh2en = TransformerModel.from_pretrained(
  '/path/to/checkpoints',
  checkpoint_file='checkpoint_best.pt',
  data_name_or_path='data-bin/wmt17_zh_en_full',
  bpe='subword_nmt',
  bpe_codes='data-bin/wmt17_zh_en_full/zh.code'
)
zh2en.translate('你好 世界')
# 'Hello World'
```

如果您使用的是`transformer.wmt19`模型，则需要将`bpe`参数设置为`'fastbpe'`并(可选)加载 4-model ensemble：
```python
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
                       checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe')
en2de.eval()  # disable dropout
```

## Example usage (CLI tools)

二分类测试集的生成可按以下方式以批次模式运行，例如适用于GTX-1080ti上的WMT 2014 English-French：
```bash
mkdir -p data-bin
curl https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf - -C data-bin
curl https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2 | tar xvjf - -C data-bin
fairseq-generate data-bin/wmt14.en-fr.newstest2014  \
    --path data-bin/wmt14.en-fr.fconv-py/model.pt \
    --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
# ...
# | Translated 3003 sentences (96311 tokens) in 166.0s (580.04 tokens/s)
# | Generate test with beam=5: BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)

# Compute BLEU score
grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
fairseq-score --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref
# BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)
```

## 训练新模型

### IWSLT'14 German to English (Transformer)

以下说明可用于训练Transformer模型在数据  [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf).

首先下载并预处理数据:
```bash
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..
# 处理后的文件结构:
examples/translation/iwslt14.tokenized.de-en
# tree iwslt14.tokenized.de-en/
iwslt14.tokenized.de-en/
├── code
├── test.de
├── test.en
├── tmp
│   ├── IWSLT14.TED.dev2010.de-en.de
│   ├── IWSLT14.TED.dev2010.de-en.en
│   ├── IWSLT14.TED.tst2010.de-en.de
│   ├── IWSLT14.TED.tst2010.de-en.en
│   ├── IWSLT14.TED.tst2011.de-en.de
│   ├── IWSLT14.TED.tst2011.de-en.en
│   ├── IWSLT14.TED.tst2012.de-en.de
│   ├── IWSLT14.TED.tst2012.de-en.en
│   ├── IWSLT14.TEDX.dev2012.de-en.de
│   ├── IWSLT14.TEDX.dev2012.de-en.en
│   ├── test.de
│   ├── test.en
│   ├── train.de
│   ├── train.en
│   ├── train.en-de
│   ├── train.tags.de-en.clean.de
│   ├── train.tags.de-en.clean.en
│   ├── train.tags.de-en.de
│   ├── train.tags.de-en.en
│   ├── train.tags.de-en.tok.de
│   ├── train.tags.de-en.tok.en
│   ├── valid.de
│   └── valid.en
├── train.de
├── train.en
├── valid.de
└── valid.en


tree orig/
orig/
├── de-en
│   ├── IWSLT14.TED.dev2010.de-en.de.xml
│   ├── IWSLT14.TED.dev2010.de-en.en.xml
│   ├── IWSLT14.TED.tst2010.de-en.de.xml
│   ├── IWSLT14.TED.tst2010.de-en.en.xml
│   ├── IWSLT14.TED.tst2011.de-en.de.xml
│   ├── IWSLT14.TED.tst2011.de-en.en.xml
│   ├── IWSLT14.TED.tst2012.de-en.de.xml
│   ├── IWSLT14.TED.tst2012.de-en.en.xml
│   ├── IWSLT14.TEDX.dev2012.de-en.de.xml
│   ├── IWSLT14.TEDX.dev2012.de-en.en.xml
│   ├── README
│   ├── train.en
│   ├── train.tags.de-en.de
│   └── train.tags.de-en.en
└── de-en.tgz


ls data-bin/
preprocess.log

# Preprocess/binarize the data, 在fairseq目录下运行
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
#处理完成后，在 fairseq目录下生成data-bin目录
tree data-bin/
data-bin/
└── iwslt14.tokenized.de-en
    ├── dict.de.txt
    ├── dict.en.txt
    ├── preprocess.log
    ├── test.de-en.de.bin
    ├── test.de-en.de.idx
    ├── test.de-en.en.bin
    ├── test.de-en.en.idx
    ├── train.de-en.de.bin
    ├── train.de-en.de.idx
    ├── train.de-en.en.bin
    ├── train.de-en.en.idx
    ├── valid.de-en.de.bin
    ├── valid.de-en.de.idx
    ├── valid.de-en.en.bin
    └── valid.de-en.en.idx
1 directory, 15 files
```

接下来，我们将针对此数据训练一个Transformer翻译模型:
```bash
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```

最后，我们可以评估我们训练过的模型:
```bash
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
```

### WMT'14 English to German (Convolutional)

以下说明可用于在WMT英语到德语数据集上训练卷积翻译模型。
See the [Scaling NMT README](../scaling_nmt/README.md) 有关在此数据上训练Transformer转换模型的说明。

可以使用“ prepare-wmt14en2de.sh”脚本对WMT英语到德语数据集进行预处理。
默认情况下，它将生成在以下情况下建模的数据集  [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), but with additional news-commentary-v12 data from WMT'17.

仅使用WMT'14中可用的数据中获得的结果，原始论文 [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](https://arxiv.org/abs/1705.03122) paper, please use the `--icml17` option.

```bash
# Download and prepare the data
cd examples/translation/
# WMT'17 data:
bash prepare-wmt14en2de.sh
# or to use WMT'14 data:
# bash prepare-wmt14en2de.sh --icml17
cd ../..

# Binarize the dataset
TEXT=examples/translation/wmt17_en_de
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

# Train the model
mkdir -p checkpoints/fconv_wmt_en_de
fairseq-train \
    data-bin/wmt17_en_de \
    --arch fconv_wmt_en_de \
    --dropout 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 4000 \
    --save-dir checkpoints/fconv_wmt_en_de

# Evaluate
fairseq-generate data-bin/wmt17_en_de \
    --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt \
    --beam 5 --remove-bpe
```

### WMT'14 English to French
```bash
# Download and prepare the data
cd examples/translation/
bash prepare-wmt14en2fr.sh
cd ../..

# Binarize the dataset
TEXT=examples/translation/wmt14_en_fr
fairseq-preprocess \
    --source-lang en --target-lang fr \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt14_en_fr --thresholdtgt 0 --thresholdsrc 0 \
    --workers 60

# Train the model
mkdir -p checkpoints/fconv_wmt_en_fr
fairseq-train \
    data-bin/wmt14_en_fr \
    --arch fconv_wmt_en_fr \
    --dropout 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 3000 \
    --save-dir checkpoints/fconv_wmt_en_fr

# Evaluate
fairseq-generate \
    data-bin/fconv_wmt_en_fr \
    --path checkpoints/fconv_wmt_en_fr/checkpoint_best.pt \
    --beam 5 --remove-bpe
```

## Multilingual Translation

我们还支持训练多语言翻译模型。在此样本中，我们将使用IWSLT'17数据集训练多语言的`{de,fr}-en`翻译模型。

请注意，此处使用的预处理与上面的IWSLT'14 En-De数据略有不同。
特别是，我们学习了所有三种语言的联合BPE代码，并使用fairseq-interactive和sacrebleu对测试集进行评分。

```bash
# First install sacrebleu and sentencepiece
pip install sacrebleu sentencepiece

# Then download and preprocess the data
cd examples/translation/
bash prepare-iwslt17-multilingual.sh
cd ../..

# Binarize the de-en dataset
TEXT=examples/translation/iwslt17.de_fr.en.bpe16k
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train.bpe.de-en \
    --validpref $TEXT/valid0.bpe.de-en,$TEXT/valid1.bpe.de-en,$TEXT/valid2.bpe.de-en,$TEXT/valid3.bpe.de-en,$TEXT/valid4.bpe.de-en,$TEXT/valid5.bpe.de-en \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10

# Binarize the fr-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
fairseq-preprocess --source-lang fr --target-lang en \
    --trainpref $TEXT/train.bpe.fr-en \
    --validpref $TEXT/valid0.bpe.fr-en,$TEXT/valid1.bpe.fr-en,$TEXT/valid2.bpe.fr-en,$TEXT/valid3.bpe.fr-en,$TEXT/valid4.bpe.fr-en,$TEXT/valid5.bpe.fr-en \
    --tgtdict data-bin/iwslt17.de_fr.en.bpe16k/dict.en.txt \
    --destdir data-bin/iwslt17.de_fr.en.bpe16k \
    --workers 10

# Train a multilingual transformer model
# NOTE: the command below assumes 1 GPU, but accumulates gradients from
#       8 fwd/bwd passes to simulate training on 8 GPUs
mkdir -p checkpoints/multilingual_transformer
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt17.de_fr.en.bpe16k/ \
    --max-epoch 50 \
    --ddp-backend=no_c10d \
    --task multilingual_translation --lang-pairs de-en,fr-en \
    --arch multilingual_transformer_iwslt_de_en \
    --share-decoders --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
    --dropout 0.3 --weight-decay 0.0001 \
    --save-dir checkpoints/multilingual_transformer \
    --max-tokens 4000 \
    --update-freq 8

# Generate and score the test set with sacrebleu
SRC=de
sacrebleu --test-set iwslt17 --language-pair ${SRC}-en --echo src \
    | python scripts/spm_encode.py --model examples/translation/iwslt17.de_fr.en.bpe16k/sentencepiece.bpe.model \
    > iwslt17.test.${SRC}-en.${SRC}.bpe
cat iwslt17.test.${SRC}-en.${SRC}.bpe \
    | fairseq-interactive data-bin/iwslt17.de_fr.en.bpe16k/ \
      --task multilingual_translation --lang-pairs de-en,fr-en \
      --source-lang ${SRC} --target-lang en \
      --path checkpoints/multilingual_transformer/checkpoint_best.pt \
      --buffer-size 2000 --batch-size 128 \
      --beam 5 --remove-bpe=sentencepiece \
    > iwslt17.test.${SRC}-en.en.sys
grep ^H iwslt17.test.${SRC}-en.en.sys | cut -f3 \
    | sacrebleu --test-set iwslt17 --language-pair ${SRC}-en
```

##### 推理期间的参数格式

在推理过程中，需要指定单个`--source-lang`和`--target-lang`，以指示推理语言的方向。 
“ --lang-pairs”，“-encoder-langtok”，“-decoder-langtok”必须设置为与training相同的值。
