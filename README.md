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

The SimpleQuestions (v2) dataset and FB2M can be downloaded [here](https://research.fb.com/downloads/babi/), the Reddit dialogue dataset associated with ConceptNet can be downloaded [here](http://coai.cs.tsinghua.edu.cn/hml/dataset/#commonsense).

## Quickstart
* Data Processing

    Please refer to the following shell scripts:
    ```
    sh gen_linkings.sh
    sh gen_candidates.sh
    sh gen_schema_dataset.sh
    sh gen_retrieving.sh
    sh gen_final_dataset.sh
    ```
* KBQA Pre-training

    Please refer to the following shell script:
    ```
    sh train_kbqa.sh
    ```
* Dialogue Generation

    Please refer to the following shell scripts:
    ```
    sh run_train.sh
    sh run_test.sh
    ```