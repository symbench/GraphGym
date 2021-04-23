from graphgym.config import cfg
from deepsnap.dataset import GraphDataset
import spice_completion.datasets as datasets
import spice_completion.datasets.helpers as h
import torch
import os
import numpy as np

from graphgym.register import register_loader

def find_netlists(rootdir='.'):
    child_list = [rootdir]
    if os.path.isdir(rootdir):
        child_list = (os.path.join(rootdir, c) for c in os.listdir(rootdir))

    netlist_paths = []
    for file_or_dir in child_list:
        if os.path.isdir(file_or_dir):
            contained_paths = find_netlists(file_or_dir)
            netlist_paths.extend(contained_paths)
        elif file_or_dir.endswith('.cir') or file_or_dir.endswith('.net'):
            netlist_paths.append(file_or_dir)

    return netlist_paths

def load_dataset(format, name, dataset_dir):
    if format != 'NetlistOmitted':
        return None

    dataset_dir = '{}/{}'.format(dataset_dir, name)
    netlists = find_netlists(dataset_dir)
    if cfg.dataset.mean:
        mean = np.load(cfg.dataset.mean)
        stddev = np.load(cfg.dataset.stddev)
        dataset = datasets.omitted(netlists, min_edge_count=5, resample=cfg.dataset.resample, mean=mean, std=stddev)
    else:
        dataset = datasets.omitted(netlists, min_edge_count=5, resample=cfg.dataset.resample)

    graphs = h.to_deepsnap(dataset)

    dataset = GraphDataset(
        graphs,
        task=cfg.dataset.task,
        edge_train_mode=cfg.dataset.edge_train_mode,
        edge_message_ratio=cfg.dataset.edge_message_ratio,
        edge_negative_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio,
        resample_disjoint=cfg.dataset.resample_disjoint,
        minimum_node_per_graph=0)
    dataset._num_graph_labels = len(datasets.helpers.component_types)
    return dataset

register_loader('omitted_netlists', load_dataset)
