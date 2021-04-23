import os
import random
import numpy as np
import argparse
import torch
import sys

from graphgym.config import (cfg, assert_cfg)
from graphgym.loader import create_dataset, create_loader
from graphgym.model_builder import create_model
from graphgym.train import train
from graphgym.utils.device import auto_select_device
from graphgym.contrib.train import *

if __name__ == '__main__':
    # Load cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file path',
        required=True,
        type=str
    )
    parser.add_argument(
        '--ckpt',
        dest='ckpt_file',
        help='Checkpoint file path',
        required=True,
        type=str
    )
    parser.add_argument(
        'opts',
        help='See graphgym/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    # Load config file
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg(cfg)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    out_dir_parent = cfg.out_dir
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    auto_select_device()

    # Set learning environment
    # TODO: pass mean, std
    datasets = create_dataset()
    loaders = create_loader(datasets)
    model = create_model(datasets)
    ckpt = torch.load(args.ckpt_file)
    model.load_state_dict(ckpt['model_state'])

    total = 0
    correct = 0
    for loader in loaders:
        for batch in loader:
            batch.to(torch.device(cfg.device))
            pred, true = model(batch)
            pred_classes = pred.argmax(axis=1)
            correct += (pred_classes == true).nonzero().shape[0]
            total += true.shape[0]
    if sys.stdout.isatty():
        print('accuracy:', correct/total, f'({correct}/{total})')
    else:
        print(correct/total)
