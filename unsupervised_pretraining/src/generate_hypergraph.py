#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
============================================================
Hypergraph View Generator (+ utilities)
============================================================
File: (put your filename here, e.g., view_generator.py)

Summary
-------
Generates augmented *hypergraph views* for self-/contrastive pretraining.
Given a PyG-style hypergraph (bipartite incidence `edge_index`), it:
  1) Row-normalizes node features (`normalize_l2`)
  2) Encodes nodes with two HypergraphConv layers
  3) Aggregates node→edge features (mean over incident nodes)
  4) Samples edge states via Gumbel-Softmax (e.g., keep / neutral / mask)
  5) Combines edge- and node-level masks (via `aug_node`) to filter incidences

Main Components
---------------
• `normalize_l2(X)` : Row-wise normalization (by row-sum; legacy name).
• `delete_hyperedge(reserve_id, H)` : Remove incidences whose edge id ∉ `reserve_id`.
• `HypergraphViewGenerator(in_dim, out_dim, head=2, dropout=0.6)` :
    - 2× HypergraphConv encoder producing edge-state logits/embeddings.
    - `forward(data, args)` returns (incidence mask, filtered `edge_index`).
"""



import copy
import torch
import torch.nn as nn, torch.nn.functional as F
import numpy as np
import math
from torch_scatter import scatter
from torch_geometric.nn import HypergraphConv
from torch_geometric.utils import subgraph

from aug import aug_node

def normalize_l2(X):
    """
    Row-wise L2 normalization of feature matrix.
    Each row vector is scaled by its L1 norm (sum of entries).
    """
    rownorm = X.detach().sum(dim=1,keepdims=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X

def delete_hyperedge(reserve_id, H):
    """
        Remove hyperedges not in reserve_id.
        Args:
            reserve_id (Tensor): indices of hyperedges to keep.
            H (Tensor): incidence matrix [2, num_edges].
        Returns:
            new_H (Tensor): filtered incidence matrix.
        """
    reserve_id = reserve_id.detach().numpy()
    H_size = H.size(1)
    H = H.detach().numpy()
    new_H = [[],[]]
    for i in range(H_size):
        if H[1,i] not in reserve_id:
            new_H[0].append(H[0][i])
            new_H[1].append(H[1][i])

    new_H = torch.Tensor(new_H)
    return new_H


# HYPEREDGE ViewGenerator
class HypergraphViewGenerator(torch.nn.Module):
    """
   Module to generate augmented hypergraph views.
   Encodes node features with HypergraphConv layers,
   applies edge/node masking and sampling via Gumbel-softmax.
   """
    def __init__(self,in_dim,out_dim,head=2, dropout=0.6):
        super().__init__()
        self.encoder = nn.ModuleList([
            HypergraphConv(in_dim, in_dim // 2, dropout=dropout),
            HypergraphConv(in_dim // 2, out_dim, dropout=dropout)
        ])  # Move to device

    def forward(self, data, args):
        """
        Args:
            data: input graph data object (with x, edge_index, overlap).
            args: arguments (must include .device).
        Returns:
            final_sample (Tensor): mask over selected edges.
            edge_index (Tensor): filtered incidence matrix.
        """
        device = args.device
        X = copy.deepcopy(data.x)  # Original features
        X = normalize_l2(X)
        edge_index = data.edge_index.long()
        
        # Augmentation for edges based on overlap
        sel_mask = aug_node(data.overlap, args, device)

        # Encode node features via hypergraph convolutions
        for m in self.encoder:
            X = m(X,edge_index)

        # Aggregate node features to edge features (mean over incident nodes)
        Xve = X[edge_index[0]]
        Xe = scatter(Xve, edge_index[1], dim = 0,reduce = 'mean') #|E|*3
        Xe = normalize_l2(torch.sigmoid(Xe))

        # Gumbel-softmax sampling over edge states
        sample = F.gumbel_softmax(Xe, hard=True)

        # State indices: reserve (0), ..., mask (2)
        reserve = sample[:, 0].bool()
        mask = sample[:, 2].bool()

        # Combine edge-level and node-level masks
        reserve_sample = reserve[edge_index[1]]
        mask_sample = torch.logical_and(mask[edge_index[1]], sel_mask[edge_index[0]].to(device))
        final_sample = reserve_sample | mask_sample

        # Subselect edges by final mask
        edge_index = edge_index[:, final_sample]
        return final_sample.float(), edge_index