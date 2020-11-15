# TransDG

The implementation of TransDG proposed in [Improving Knowledge-aware Dialogue Generation via Knowledge Base Question Answering](https://arxiv.org/pdf/1912.07491.pdf) (AAAI-2020). 

In this paper, we propose a novel knowledge-aware dialogue generation model (called TransDG), which transfers question representation and knowledge matching abilities from knowledge base question answering (KBQA) task to facilitate the utterance understanding and factual knowledge selection for dialogue generation.

## Requirements
* Python3
* Tensorflow >= 1.8
* Stanford CoreNLP
* NLTK
* PyLucene

## Datasets

The SimpleQuestions (v2) dataset and FB2M can be downloaded from [here](https://research.fb.com/downloads/babi/), the Reddit dialogue dataset associated with ConceptNet can be downloaded from [here](http://coai.cs.tsinghua.edu.cn/hml/dataset/#commonsense). Also, please download the Freebase metadata from [here](https://pan.baidu.com/s/1MauWDI5NY23J_5FKUOGHaw) (extraction code: iwq1) and unpack to the folder `data/freebase-metadata/`, download the Glove embeddings from [here](https://pan.baidu.com/s/1ekrkqxuLhSMfqHELxLOJnA) (extraction code: vxc3) and unpack to the folder `data/kbqa_emb/`.

## Quickstart

### Step 1: Data Processing

Data Processing for KBQA Pre-training:

(1) Entity linking. Please set `data_name=SimpQ` and `data_dir=data/SimpleQuestions_v2` in the `gen_linkings.sh`, then run:
```
sh gen_linkings.sh
```
(2) Candidates building. Please set `data_name=SimpQ` in the `gen_candidates.sh`, then run:
```
sh gen_candidates.sh
```

Data Processing for Dialogue Generation:

(1) Top-k similar responses retrieving. Please set `mode=train/valid/test` in the `gen_retrieving.sh`, then run:
```
sh gen_retrieving.sh
```

(2) Entity linking. Please set `data_name=Reddit`, `data_dir=data/Reddit`, and `mode=train/valid/test` in the `gen_linkings.sh`, then run:
```
sh gen_linkings.sh
```
(3) Candidates building. Please set `data_name=Reddit` and `mode=train/valid/test` in the `gen_candidates.sh`, then run:
```
sh gen_candidates.sh
```

(4) Schema dataset building. Please set `mode=train/valid/test` in the `gen_schema_dataset.sh`, then run:
```
sh gen_schema_dataset.sh
```

(5) Final dataset building. Please set `mode=train/valid/test` in the `gen_final_dataset.sh`, then run:
```
sh gen_final_dataset.sh
```

### Step 2: KBQA Pre-training

Please refer to the script `train_kbqa.sh` and set parameters accordingly. Then run:
```
sh train_kbqa.sh
```

### Step 3: Dialogue Generation

For model training, please refer to the script `run_train.sh` and set parameters accordingly. Then run:
```
sh run_train.sh
```
For model testing, please refer to the script `run_test.sh` and set parameters accordingly. Then run:
```
sh run_test.sh
```