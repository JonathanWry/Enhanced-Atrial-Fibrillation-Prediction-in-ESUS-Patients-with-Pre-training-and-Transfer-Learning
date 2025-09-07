#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
==========================================================================
Data Utilities: Hypergraph Construction, Normalization, and Split Helpers
==========================================================================

File
----
data_utils.py   (paste this header at the top of the file)

Purpose
-------
Utility functions for preparing hypergraph data structures and splits.

  • ExtractV2E(data)
      Ensures edge_index only contains V→E incidence (removes self-loops).

  • Add_Self_Loops(data)
      Adds node-specific self-loop hyperedges for isolated nodes, updates
      edge_index and total edge count.

  • norm_contruction(data, option, TYPE)
      Builds normalization weights for incidence (V2E) or adjacency (V2V):
         - 'all_one'       : uniform weights
         - 'deg_half_sym'  : symmetric degree-based normalization

  • rand_train_test_idx(label, train_prop, valid_prop, …)
      Splits labeled data into train/valid/test indices. Supports:
         - ignoring negative labels
         - class-balanced splits
         - reproducibility with rand_seed

"""

import torch

import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def ExtractV2E(data):
    """
    Extract only the V→E incidence portion of a bipartite hypergraph.

    Assumes:
        data.edge_index = stacked [V|E ; E|V]

    Sorts edge_index by node indices, trims off E→V half, and updates data.

    Args:
        data: PyG Data object with .edge_index, .n_x, .num_hyperedges.

    Returns:
        Updated data with edge_index containing only V→E edges.
    """
    # Assume edge_index = [V|E;E|V]
    edge_index = data.edge_index
    #     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges
    cidx = torch.where(edge_index[0] == num_nodes)[
        0].min()  # cidx: [V...|cidx E...]
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    return data


def Add_Self_Loops(data):
    """
    Add per-node self-loop hyperedges for nodes without them.

    - Uses current incidence edge_index (V,E).
    - Creates new unique hyperedge IDs for missing self-loops.
    - Updates .totedges and re-sorts edge_index.

    Args:
        data: PyG Data object with .edge_index, .n_x, .num_hyperedges.

    Returns:
        Updated data with appended self-loop edges.
    """
    # update so we dont jump on some indices
    # Assume edge_index = [V;E]. If not, use ExtractV2E()
    edge_index = data.edge_index
    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges


    hyperedge_appear_fre = Counter(edge_index[1].numpy())
    # store the nodes that already have self-loops
    skip_node_set = set()
    # skip_node_set = []
    for edge in hyperedge_appear_fre:
        if hyperedge_appear_fre[edge] == 1:
            skip_node = edge_index[0][torch.where(
                edge_index[1] == edge)[0].item()]
            # skip_node_lst.append(skip_node.item())
            skip_node_set.add(skip_node.item())


    num_new_edges = num_nodes - len(skip_node_set)

    # Initialize new_edges tensor with correct size
    new_edges = torch.zeros((2, num_new_edges), dtype=edge_index.dtype)
    tmp_count = 0
    new_edge_idx = edge_index[1].max() + 1
    for i in range(num_nodes):
        if i not in skip_node_set:
            if tmp_count >= num_new_edges:  # Sanity check
                raise RuntimeError(f"tmp_count ({tmp_count}) exceeds num_new_edges ({num_new_edges})")
            new_edges[0][tmp_count] = i
            new_edges[1][tmp_count] = new_edge_idx
            new_edge_idx += 1
            tmp_count += 1

    data.totedges = num_hyperedges + num_nodes - len(skip_node_set)
    edge_index = torch.cat((edge_index, new_edges), dim=1)
    # Sort along w.r.t. nodes
    _, sorted_idx = torch.sort(edge_index[0])
    data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    return data


def norm_contruction(data, option='all_one', TYPE='V2E'):
    """
    Construct normalization factors for incidence or V2V graph.

    Options:
        option='all_one'       → all weights = 1
        option='deg_half_sym'  → symmetric degree normalization

    TYPE:
        'V2E' → bipartite incidence normalization
        'V2V' → gcn_norm applied to V2V projection

    Args:
        data: PyG Data with .edge_index.
        option (str): normalization scheme.
        TYPE (str): 'V2E' or 'V2V'.

    Returns:
        Updated data with .norm tensor.
    """
    if TYPE == 'V2E':
        if option == 'all_one':
            data.norm = torch.ones_like(data.edge_index[0])

        elif option == 'deg_half_sym':
            edge_weight = torch.ones_like(data.edge_index[0])
            cidx = data.edge_index[1].min()
            Vdeg = scatter_add(edge_weight, data.edge_index[0], dim=0)
            HEdeg = scatter_add(edge_weight, data.edge_index[1] - cidx, dim=0)
            V_norm = Vdeg ** (-1 / 2)
            E_norm = HEdeg ** (-1 / 2)
            data.norm = V_norm[data.edge_index[0]] * \
                        E_norm[data.edge_index[1] - cidx]

    elif TYPE == 'V2V':
        data.edge_index, data.norm = gcn_norm(
            data.edge_index, data.norm, add_self_loops=True)
    return data


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=False, balance=False, rand_seed=0):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)
        np.random.seed(rand_seed)

        perm = torch.as_tensor(np.random.permutation(n))
        print(perm)

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            split_idx = {'train': train_indices,
                         'valid': val_indices,
                         'test': test_indices}
            return split_idx  # HERE

        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]

        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    else:
        indices = []
        for i in range(label.max() + 1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop / (label.max() + 1) * len(label))
        val_lb = int(valid_prop * len(label))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    return split_idx