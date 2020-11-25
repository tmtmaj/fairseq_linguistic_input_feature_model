# Introduction

This repository is to reproduce _Linguistic Input Features Improve Neural Machine Translation_ in fairseq-seq.

# Requirements and Installation

PyTorch version >= 1.4.0  
Python version >= 3.6

```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

Implemention env 

Python Python 3.6.9 on linux  
Pytorch 1.7.0+cu101

# Tutorial 

## Data preprocess

we introduce the Korean-to-English translation where the NMT input is augmented with part-of-speech(POS) tags.

First, you should prepare your tokenized&bped file and create linguistic annotation input files like:

```
train.ko train.en valid.ko valid.en test.ko test.en
train.feature valid.feature test.feature
```

```
head -n 3 train.ko

올해 87 차 를 맞은 sc@@ e 수련 회 는 매년 전국 교회 에서 다음 세대 성도 들 이 한자리 에 모이는 여름철 대표 신앙 축제 다 .
이 달 들어 미국 과 중국 이 상대 국 제품 에 고 율 의 관세 를 부과 하는 등 무역 전쟁 이 격@@ 해지면서 한국 의 경기 전망 이 불투명@@ 해졌다 .
왜 내 가 하는 일 은 노동 이 아니@@ 거나 대가 가 없는지 , 지금 까지 하던 대로 싸@@ 우면 이 세상 이 조금 이라도 나아@@ 질지 , 답답한 누구 라도 청계 광장 으로 나가 자기 목소리 를 내@@ 도 좋겠다 .

```

```
head -n 3 train.feature

Noun Number Noun Josa Verb Alpha Alpha Noun Noun Josa Noun Noun Noun Josa Noun Noun Noun Suffix Josa Noun Josa Verb Noun Noun Noun Noun Josa Punctuation
Determiner Noun Verb Noun Josa Noun Josa Noun Noun Noun Josa Modifier Noun Josa Noun Josa Noun Verb Noun Noun Noun Josa Adjective Adjective Noun Josa Noun Noun Josa Adjective Adjective Punctuation
Noun Noun Josa Verb Noun Josa Noun Josa Adjective Adjective Noun Josa Adjective Punctuation Noun Josa Verb Noun Verb Verb Noun Noun Josa Noun Josa Verb Verb Punctuation Adjective Noun Josa Noun Noun Josa Verb Noun Noun Josa Verb Verb Adjective Punctuation

```


Then preprocess data using fairseq-py:

```
python /fairseq_linguistic_input_feature_model/fairseq-master/fairseq_cli/preprocess.py \
    --source-lang ko --target-lang en \
    --trainpref /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/train \
    --validpref /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/valid \
    --testpref /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/test \
    --feature-suffix feature \
    --destdir /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/fairseq_prepro_bpe \
    --workers 32

```

## Model Train

train model using fairseq-py

```
python /fairseq_linguistic_input_feature_model/fairseq-master/fairseq_cli/train.py \
     /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/fairseq_prepro_bpe \
    --arch transformer --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --activation-fn relu\
    --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 --max-tokens 3500 --label-smoothing 0.1 \
    --save-dir  /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/  --log-interval 200 --max-epoch 30 \
    --keep-interval-updates -1 --save-interval-updates 0 --criterion label_smoothed_cross_entropy --update-freq 8 \
    --load-features  --feature-merge "concat" \
    | tee -a  /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/training.log \
```

## Translate

```
python /fairseq_linguistic_input_feature_model/fairseq-master/fairseq_cli/generate.py \
    /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/fairseq_prepro_bpe \
     --gen-subset test  --load-features \
    --source-lang ko --target-lang en \
    --path /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/checkpoint_last.pt --beam 5 --nbest 1 --quiet 
```

The important options we add:

```
parser.add_argument('--feature-merge', type=str, metavar='STR', default=None, help='feature merging method')
parser.add_argument('--load-features', action='store_true',help='load the binarized alignments')                           
group.add_argument("--feature-suffix", metavar="FP", default=None, help="feature file suffix")
```

`--feature-merge` Merge action for incorporating linguistic input features embeddings.  
Possible choices: concat, add, gate  Default: None

## Train a vanilla NMT model

```
python /fairseq_linguistic_input_feature_model/fairseq-master/fairseq_cli/preprocess.py \
    --source-lang ko --target-lang en \
    --trainpref /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/train \
    --validpref /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/valid \
    --testpref /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/test \
    --destdir /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/fairseq_prepro_bpe \
    --workers 32
    
python /fairseq_linguistic_input_feature_model/fairseq-master/fairseq_cli/train.py \
     /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/fairseq_prepro_bpe \
    --arch transformer --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --activation-fn relu\
    --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 --max-tokens 3500 --label-smoothing 0.1 \
    --save-dir  /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/  --log-interval 200 --max-epoch 30 \
    --keep-interval-updates -1 --save-interval-updates 0 --criterion label_smoothed_cross_entropy --update-freq 8 \
    | tee -a  /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/training.log \
    
python /fairseq_linguistic_input_feature_model/fairseq-master/fairseq_cli/generate.py \
    /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/fairseq_prepro_bpe \
     --gen-subset test  \
    --source-lang ko --target-lang en \
    --path /fairseq_linguistic_input_feature_model/fairseq-master/tutorial_dataset/checkpoint_last.pt --beam 5 --nbest 1 --quiet 

```

# Reference

```bibtex
@inproceedings{sennrich-haddow-2016-linguistic,
    title = "Linguistic Input Features Improve Neural Machine Translation",
    author = "Sennrich, Rico  and
      Haddow, Barry",
    booktitle = "Proceedings of the First Conference on Machine Translation: Volume 1, Research Papers",
    month = aug,
    year = "2016",
    address = "Berlin, Germany",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W16-2209",
    doi = "10.18653/v1/W16-2209",
    pages = "83--91",
}

@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
