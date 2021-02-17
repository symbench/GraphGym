from deepsnap.dataset import GraphDataset
from deepsnap.graph import Graph
import spice_completion.datasets as datasets
import torch
import os

from graphgym.register import register_loader

def find_netlists(rootdir='.'):
    netlist_paths = []
    for file_or_dir in (os.path.join(rootdir, c) for c in os.listdir(rootdir)):
        if os.path.isdir(file_or_dir):
            contained_paths = find_netlists(file_or_dir)
            netlist_paths.extend(contained_paths)
        elif file_or_dir.endswith('.cir') or file_or_dir.endswith('.net'):
            netlist_paths.append(file_or_dir)

    return netlist_paths

def ensure_no_nan(tensor):
    nan_idx = torch.isnan(tensor).nonzero(as_tuple=True)
    nan_count = nan_idx[0].shape[0]
    assert nan_count == 0, 'nodes contain nans'

def spektral_to_deepsnap(dataset):
    graphs = []
    for sgraph in dataset:
        nxgraph = dataset.to_networkx(sgraph)
        label = torch.tensor([sgraph.y.argmax()])
        node_features = torch.tensor(sgraph.x)
        ensure_no_nan(node_features)

        Graph.add_graph_attr(nxgraph, 'graph_label', label)
        Graph.add_graph_attr(nxgraph, 'node_feature', node_features)
        graphs.append(Graph(nxgraph))

    return graphs

def load_dataset(format, name, dataset_dir):
    if format != 'NetlistOmitted':
        return None

    print('we are loading a dataset...?', format, name, dataset_dir)
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    netlists = find_netlists(dataset_dir)
    print('netlists:', netlists)
    dataset = datasets.omitted(netlists)
    graphs = spektral_to_deepsnap(dataset)
    return graphs

register_loader('omitted_netlists', load_dataset)
