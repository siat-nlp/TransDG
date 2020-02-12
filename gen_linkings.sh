#!/bin/bash
 
# set python path
python_path='python3 -m'

# set arguments
data_name=Reddit   # ["SimpQ", "Reddit"]
mode=test        # only for Reddit dataset, mode=["train", "valid", "test"]
data_dir=./data/SimpleQuestions_v2
#data_dir=./data/Reddit

fb_dir=./data/SimpleQuestions_v2/freebase-subsets
fb_meta_dir=./data/freebase-metadata

${python_path} src.kbqa.scripts.gen_linkings --data_name=${data_name} --mode=${mode} --data_dir=${data_dir} --fb_dir=${fb_dir} --fb_meta_dir=${fb_meta_dir}
