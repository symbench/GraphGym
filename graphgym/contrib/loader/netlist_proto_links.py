from graphgym.config import cfg
from deepsnap.dataset import GraphDataset
from spice_completion.datasets import PrototypeLinkDataset
import spice_completion.datasets.helpers as h
import torch
import os

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
    if format != 'NetlistProtoLinks':
        return None

    dataset_dir = '{}/{}'.format(dataset_dir, name)
    netlists = find_netlists(dataset_dir)
    dataset = PrototypeLinkDataset(netlists, normalize=False)
    graphs = h.to_deepsnap(dataset)

    return graphs

register_loader('netlist_proto_links', load_dataset)
