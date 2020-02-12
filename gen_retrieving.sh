#!/bin/bash

# set python path
python_path='python3 -m'

# set arguments
top_k=3

${python_path} src.utils.index_builder

${python_path} src.utils.response_candgen --top_k=${top_k}
