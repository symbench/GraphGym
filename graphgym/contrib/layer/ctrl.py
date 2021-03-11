"""
This layer implements a graph embedding with roots in control theory. There
are no learnable parameters. It also requires the following options in the cfg:

"""
from graphgym.config import cfg
from graphgym.register import register_layer
import torch.nn as nn
import torch

import controlpy
import networkx as nx
import numpy as np
import numpy.linalg as LA
import pandas as pd

class InvalidConfigException(Exception):
    def __init__(self, name, value):
        message = f'{name} can only be {value} when using the CTRL layer'
        super(InvalidConfigException, self).__init__(message)

#default parameters
INCLUDE_GRM = True
INCLUDE_INVERSE_GRM = False 
INCLUDE_EIGEN_VALUES = True
INCLUDE_MIN_MAX = True
INCLUDE_MEAN = True
INCLUDE_LAP_SPECT = True
INCLUDE_LAP_SPECT_EXT = True
INCLUDE_METRIC_DIMENSION = True
INCLUDE_FEATURES = True
LAPLACIAN_ENERGY = True
ECC_SPECTRUM = True 
ECC_ENERGY = True
WEINER_INDEX = True
TRACE_DEG_SEQ = True
INCLUDE_CYCLES = True

def get_descriptor(G, num_iterations = 30):
    G = nx.convert_node_labels_to_integers(G)
    if G.number_of_nodes() < 10:
        G = add_multiple_copies(G)
    A1 = np.matrix(nx.laplacian_matrix(G,nodelist=sorted(G.nodes())).todense())
    n = A1.shape[0]
    desc = {}
    leaders = []
    for num_l_idx, num_leaders in enumerate([1,2,5,9,int(n*2/100)+1,int(n*5/100)+1,int(n*10/100)+1,int(n*20/100)+1,int(n*30/100)+1]):
        old_n = A1.shape[0]
        traces = []
        ranks = []
        min_eigs = []
        max_eigs = []
        itraces = []
        iranks = []
        imin_eigs = []
        imax_eigs = []
        metric_dimension = []
        for i in range(num_iterations):
            follows = np.random.choice(old_n,old_n-num_leaders,replace=False)
            leaders = list(set(range(old_n))-set(follows))
            A2 = A1[follows, :]
            A3 = A2[:, follows]
            A = -1*A3
            n = A.shape[0]
            mask = np.ones(old_n, dtype=bool)
            mask[leaders] = False
            B = A1[mask,:] 
            B = B[:,leaders] 
            A = np.mat(A)
            B = np.mat(B)
            if INCLUDE_GRM:
                grm = controlpy.analysis.controllability_gramian(A,B)
                rnk = LA.matrix_rank(grm)
                ranks.append(rnk)
                traces.append( np.trace(grm) )
                if INCLUDE_EIGEN_VALUES :
                    w, v = LA.eig(grm)
                    a = np.real(w)
                    a[a == 0] = 0.0001
                    minval = np.min(a)
                    min_eigs.append( np.min( minval ) )
                    max_eigs.append(np.max( a ) )
                if INCLUDE_INVERSE_GRM :
                    try:
                        grm = LA.pinv( grm ,hermitian=True)
                    except np.linalg.LinAlgError:
                        return None
                    itraces.append(np.trace(grm))
                    iranks.append( LA.matrix_rank(grm) )
                    if INCLUDE_EIGEN_VALUES :
                        w, v = LA.eig(grm)
                        a = np.real(w)
                        a[a == 0] = 0.0001
                        minval = np.min(a)
                        imin_eigs.append( np.min( minval ) )
                        imax_eigs.append( np.max( a ) )  
                        
            if INCLUDE_METRIC_DIMENSION:            
                metric_dimension.append(compute_metric_dimension(G, leaders))
                                        
            
        if INCLUDE_MIN_MAX :
            desc['GRM_MAX_TRACE_'+str(num_l_idx)] = np.max(traces)
            desc['GRM_MIN_TRACE_'+str(num_l_idx)] = np.min(traces) 
            desc['GRM_MAX_RANK_'+str(num_l_idx)]  = np.max(ranks) 
            desc['GRM_MIN_RANK_'+str(num_l_idx)]  = np.min(ranks) 
            if INCLUDE_EIGEN_VALUES :

                desc['GRM_MAX_of_MIN_EIG_'+str(num_l_idx)] = np.max(min_eigs)
                desc['GRM_MIN_of_MIN_EIG_'+str(num_l_idx)] = np.min(min_eigs)

                desc['GRM_MAX_of_MAX_EIG_'+str(num_l_idx)] = np.max(max_eigs)
                desc['GRM_MIN_of_MAX_EIG_'+str(num_l_idx)] = np.min(max_eigs)
            if INCLUDE_INVERSE_GRM :    

                desc['INV_GRM_MAX_TRACE_'+str(num_l_idx)] = np.max(itraces)
                desc['INV_GRM_MIN_TRACE_'+str(num_l_idx)] = np.min(itraces)
        
        if INCLUDE_MEAN :
            desc['GRM_MEAN_TRACE_'+str(num_l_idx)] = np.mean(traces)
            desc['GRM_MEAN_RANK_'+str(num_l_idx)] = np.mean(ranks)

            if INCLUDE_EIGEN_VALUES :
                desc['GRM_MEAN_MAX_EIG_'+str(num_l_idx)] = np.mean(max_eigs)
                desc['GRM_MEAN_MIN_EIG_'+str(num_l_idx)] = np.mean(min_eigs)
                
            if INCLUDE_INVERSE_GRM :
                desc['INV_GRM_MEAN_TRACE_'+str(num_l_idx)] = np.mean(itraces)
                desc['INV_GRM_MEAN_RANK_'+str(num_l_idx)] = np.mean(iranks)
                
                if INCLUDE_EIGEN_VALUES :
                    desc['INV_GRM_MEAN_MIN_EIG_'+str(num_l_idx)] = np.mean(imin_eigs)
                    desc['INV_GRM_MEAN_MAX_EIG_'+str(num_l_idx)] =np.mean(imax_eigs)
        
        if INCLUDE_METRIC_DIMENSION:
            desc['METRIC_DIMENSION_MEAN'+str(num_l_idx)] = np.mean(metric_dimension)
            desc['METRIC_DIMENSION_MIN'+str(num_l_idx)] = np.min(metric_dimension)
            desc['METRIC_DIMENSION_MAX'+str(num_l_idx)] = np.max(metric_dimension)
    
    n,m = G.number_of_nodes(),G.number_of_edges()
    desc['no_nodes'] = n
    desc['no_edges'] = m
    desc['no_bi_conn_comp'] =len(list(nx.biconnected_components(G)))
    if INCLUDE_LAP_SPECT :
        Ls = sorted(nx.laplacian_spectrum(G))
#         desc['LS_0'] = Ls[0]
        desc['LS_1'] = Ls[1]
        desc['LS_2'] = Ls[2]
    if INCLUDE_LAP_SPECT_EXT :
        desc['LSE_3'] = Ls[3]
        desc['LSE_4'] = Ls[4]
        desc['LSE_-1'] = Ls[-1]
        desc['LSE_-2'] = Ls[-2]
        desc['LSE_-3'] = Ls[-3]
    if INCLUDE_CYCLES:
        if n-m == 1:
            desc['0cyc'] = 1  
        else:
            desc['0cyc'] = 0
        if n-m == 0:
            desc['1cyc'] = 1
        else:
            desc['1cyc'] = 0
        if n-m == -1:
            desc['2cyc'] = 1
        else:
            desc['2cyc'] = 0
        if n-m < -1:
            desc['g2cyc'] = 1
        else:
            desc['g2cyc'] = 0

    if INCLUDE_FEATURES:
        desc = {**desc, **add_features(G)} 
    return(desc)
    

def add_multiple_copies(G):
    no_node = G.number_of_nodes()   
    H = G.copy()
    H = nx.convert_node_labels_to_integers(H, ordering='sorted')
    val = sorted(G.nodes())
    val0 = val[0]
    while H.number_of_nodes()<10:
        val1 = H.number_of_nodes()-1
        H = nx.union(H, G, rename=('G1-', 'G2-'))
        H.add_edge('G1-'+str(val1), 'G2-'+str(val0))
        H = nx.convert_node_labels_to_integers(H, ordering='sorted')
    return H

def trace_deq_seq(G):
    seq = sorted(list(dict(nx.degree(G)).values()),reverse = False)
    trc = 0
    for idx, i in enumerate(seq):
        if idx+1<=i:
            trc = idx+1
    return trc

def add_features(G):
    L = nx.laplacian_matrix(G).todense()
    eig = LA.eigvals(L)
    avg_deg = 2*(G.number_of_edges()/G.number_of_nodes()) 
    lap_energy = sum([abs(i-avg_deg) for i in eig])
    dist_matrix =np.array(nx.floyd_warshall_numpy(G,nodelist=sorted(G.nodes()))) 
    eccentricity = dist_matrix * (dist_matrix >= np.sort(dist_matrix, axis=1)[:,[-1]]).astype(int) 
    e_vals = LA.eigvals(eccentricity) 
    largest_eig = np.real(max(e_vals))
    energy = np.real(sum([abs(x) for x in e_vals]))
    wiener_index = nx.wiener_index(G)
    trace_DS = trace_deq_seq(G)  
    return {'lap_energy':lap_energy,'ecc_spectrum':largest_eig,'ecc_energy':energy,'wiener_index':wiener_index,'trace_deg_seq':trace_DS}

def compute_metric_dimension(G, leaders):
    nodes_list = list(G.nodes()) 
    distance_vec = []
    for n in nodes_list:
        v_ = [nx.shortest_path_length(G, source=n, target=l) for l in leaders]
        distance_vec.append(v_)
    distance_vec = np.array(distance_vec)
    return np.unique(distance_vec, axis=0).shape[0]

        
def add_vertex(G):
    G = nx.convert_node_labels_to_integers(G)
    nodes = sorted(list(G.nodes()))
    new_vertex = nodes[-1]+1 
    G.add_node(new_vertex)
    for n in G.nodes():
        if n !=new_vertex:
            G.add_edge(new_vertex, n)
    return G
    
def get_embedding(G, node_count, device):
    descriptor = get_descriptor(G, cfg.gnn.ctrl_iterations)
    return torch.tensor(list(descriptor.values()), dtype=torch.float, device=device).expand(node_count, -1)

def check_config_compat(cfg):
    assert cfg.gnn.layers_pre_mp == 0, InvalidConfigException('cfg.gnn.layers_pre_mp', 0)
    assert cfg.gnn.layers_mp == 1, InvalidConfigException('cfg.gnn.layers_mp', 1)
    assert cfg.model.graph_pooling == 'mean', InvalidConfigException('cfg.model.graph_pooling', 'mean')

class Ctrl(nn.Module):
    r"""Implementation of CTRL+ layer (non-learning)

    """

    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(Ctrl, self).__init__(**kwargs)
        check_config_compat(cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, batch):
        """Set all the nodes to the graph embedding"""
        if not hasattr(batch.G[0], 'cached_ctrl_embedding'):
            device = batch.node_feature.device

            for (i, G) in enumerate(batch.G):
                count = G.number_of_nodes()
                if not nx.is_connected(G):
                    G = add_vertex(G)
                setattr(batch.G[i], 'cached_ctrl_embedding', get_embedding(G, count, device))

        graph_embeddings = [ G.cached_ctrl_embedding for G in batch.G ]
        batch.node_feature = torch.cat(graph_embeddings)

        return batch

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# Remember to register your layer!
register_layer('ctrl', Ctrl)
