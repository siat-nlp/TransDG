#!/bin/bash

# set python path
python_path='python3 -m'

# set arguments
data_dir=./data/Reddit
mode=test    # ['train', 'valid', 'test']
candgen_dir=./runnings/candgen_Reddit_${mode}

${python_path} src.kbqa.scripts.gen_schema_dataset --data_dir=${data_dir} --candgen_dir=${candgen_dir} --mode=${mode}
