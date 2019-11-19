#!/bin/bash

# set python path
export CUDA_VISIBLE_DEVICES=0
python_path='python3 -m'

# data config arguments
data_dir=./data/final_data
kd_dir=./data/Reddit
model_dir=./runnings/SimpQ_fc512
log_dir=./log_mlp
ckpt=best.model

# model config arguments
cell_class=GRU         # ['RNN', 'GRU', 'LSTM']
num_units=512
num_layers=2
vocab_size=30000
max_dec_len=60

# training config arguments
batch_size=100
max_epoch=20
lr_rate=0.0001
max_grad_norm=5.0
save_per_step=1000

${python_path} src.main --mode=test \
    --data_dir=${data_dir} \
    --kd_dir=${kd_dir} \
    --model_dir=${model_dir} \
    --log_dir=${log_dir} \
    --ckpt=${ckpt} \
    --cell_class=${cell_class} \
    --num_units=${num_units} \
    --num_layers=${num_layers} \
    --vocab_size=${vocab_size} \
    --max_dec_len=${max_dec_len} \
    --batch_size=${batch_size} \
    --max_epoch=${max_epoch} \
    --lr_rate=${lr_rate} \
    --max_grad_norm=${max_grad_norm} \
    --save_per_step=${save_per_step}
