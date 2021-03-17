#!/usr/bin/env bash

python -m pip install -r requirements.txt
python -m pip install git+https://github.com/snap-stanford/deepsnap.git
python -m pip install git+https://github.com/symbench/spice-completion.git
python -m pip install torch-scatter git+https://github.com/rusty1s/pytorch_sparse.git -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
python -m pip install git+git://github.com/symbench/spice-completion.git
