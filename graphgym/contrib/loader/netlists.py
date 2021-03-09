from graphgym.config import cfg
from deepsnap.dataset import GraphDataset
from deepsnap.graph import Graph
import spice_completion.datasets as datasets
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
        # TODO: Check the number of edges
        graphs.append(Graph(nxgraph))

    return graphs

def load_dataset(format, name, dataset_dir):
    if format != 'NetlistOmitted':
        return None

    dataset_dir = '{}/{}'.format(dataset_dir, name)
    netlists = find_netlists(dataset_dir)
    dataset = datasets.omitted(netlists, min_edge_count=5)
    graphs = spektral_to_deepsnap(dataset)

    dataset = GraphDataset(
        graphs,
        task=cfg.dataset.task,
        edge_train_mode=cfg.dataset.edge_train_mode,
        edge_message_ratio=cfg.dataset.edge_message_ratio,
        edge_negative_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio,
        minimum_node_per_graph=0)
    dataset._num_graph_labels = len(datasets.helpers.component_types)
    return dataset

register_loader('omitted_netlists', load_dataset)
