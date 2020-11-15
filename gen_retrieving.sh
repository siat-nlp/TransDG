#!/bin/bash

# set python path
python_path='python3 -m'

# set arguments
data_dir=./data/Reddit
index_dir=./index
mode=test         # only for Reddit dataset, mode=["train", "valid", "test"]
top_k=3

${python_path} src.scripts.index_builder --data_dir=${data_dir} --index_dir=${index_dir}

${python_path} src.scripts.response_candgen --data_dir=${data_dir} --index_dir=${index_dir} --mode=${mode} --top_k=${top_k}
