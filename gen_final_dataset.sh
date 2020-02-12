#!/bin/bash

# set python path
python_path='python3 -m'

# set arguments
mode=train      # ["train", "valid", "test"]
data_dir=./data/Reddit
candgen_dir=./runnings/candgen_Reddit_${mode}
dict_path=./runnings/candgen_SimpQ/active_dicts.pkl
save_dir=./data/final_data

${python_path} src.scripts.preprocess_dataset --mode=${mode} --data_dir=${data_dir} --candgen_dir=${candgen_dir} --dict_path=${dict_path} --save_dir=${save_dir}
