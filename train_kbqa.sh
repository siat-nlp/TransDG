#!/bin/bash

# set python path
export CUDA_VISIBLE_DEVICES=3
python_path='python3 -m'

# data config arguments
data_dir=./data/SimpleQuestions_v2
candgen_dir=./runnings/candgen_SimpQ
emb_dir=./data/kbqa_emb
fb_meta_dir=./data/freebase-metadata
output_dir=./runnings/${data_name}

# schema config arguments
qw_max_len=20
pw_max_len=8
path_max_size=3
pseq_max_len=3

# negative sampling arguments
neg_f1_ths=0.1
neg_max_sample=20      # [10, 20, 40, 80]
neg_strategy=Fix       # ['Fix', 'Dyn']

# model config arguments
cell_class=GRU         # ['RNN', 'GRU', 'LSTM']
num_units=150
num_layers=1
dim_emb=300
n_words=61814
n_mids=3561
n_paths=3561
w_emb_fix=Upd          # ['Upd', 'Fix']
drop_rate=0.2
final_func=bilinear        # [bilinear, fcxxx]

# training config arguments
loss_margin=0.5
lr_rate=0.001
optm_batch_size=128
eval_batch_size=512
max_epoch=20


${python_path} src.kbqa.main_kbqa \
    --data_dir=${data_dir} \
    --candgen_dir=${candgen_dir} \
    --emb_dir=${emb_dir} \
    --fb_meta_dir=${fb_meta_dir} \
    --output_dir=${output_dir} \
    --qw_max_len=${qw_max_len} \
    --pw_max_len=${pw_max_len} \
    --path_max_size=${path_max_size} \
    --pseq_max_len=${pseq_max_len} \
    --neg_f1_ths=${neg_f1_ths} \
    --neg_max_sample=${neg_max_sample} \
    --neg_strategy=${neg_strategy} \
    --cell_class=${cell_class} \
    --num_units=${num_units} \
    --num_layers=${num_layers} \
    --dim_emb=${dim_emb} \
    --n_words=${n_words} \
    --n_mids=${n_mids} \
    --n_paths=${n_paths} \
    --w_emb_fix=${w_emb_fix} \
    --drop_rate=${drop_rate} \
    --final_func=${final_func} \
    --loss_margin=${loss_margin} \
    --lr_rate=${lr_rate} \
    --optm_batch_size=${optm_batch_size} \
    --eval_batch_size=${eval_batch_size} \
    --max_epoch=${max_epoch}
