#!/usr/bin/env bash

pip install -r requirements.txt
pip install git+https://github.com/snap-stanford/deepsnap.git
pip install git+https://github.com/symbench/spice-completion.git
pip install torch-scatter git+https://github.com/rusty1s/pytorch_sparse.git -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
