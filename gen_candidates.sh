#!/bin/bash

# set python path
python_path='python3 -m'

# set arguments
data_name=Reddit   # ["SimpQ", "Reddit"]
output_dir=./runnings
freebase_dir=./data/SimpleQuestions_v2/freebase-subsets
verbose=0         # [0, 1]

if [ "${data_name}" = "SimpQ" ]; then
    data_dir=./data/SimpleQuestions_v2
    output_cand_dir=${output_dir}/candgen_SimpQ
    ${python_path} src.kbqa.scripts.simpq_candgen --data_dir=${data_dir} --freebase_dir=${freebase_dir} --output_dir=${output_cand_dir} --verbose=${verbose}
else
    data_dir=./data/Reddit
    mode=valid        # only for Reddit dataset, mode=["train", "valid", "test"]
    output_prefix=${output_dir}/candgen_Reddit
    ${python_path} src.kbqa.scripts.reddit_candgen --data_dir=${data_dir} --mode=${mode} --freebase_dir=${freebase_dir} --output_prefix=${output_prefix} --verbose=${verbose}
fi
