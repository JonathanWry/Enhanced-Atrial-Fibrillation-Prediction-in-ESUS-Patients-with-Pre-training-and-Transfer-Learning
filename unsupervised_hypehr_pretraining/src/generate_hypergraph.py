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
    """Row-normalize  matrix"""
    rownorm = X.detach().sum(dim=1,keepdims=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X

def delete_hyperedge(reserve_id, H):
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
    def __init__(self,in_dim,out_dim,head=2, dropout=0.6):
        super().__init__()
        # self.encoder = nn.ModuleList([HypergraphConv(in_dim, in_dim//2, head=head, droupout=dropout), HypergraphConv(in_dim//2,out_dim,head=head,dropout=dropout)])
        self.encoder = nn.ModuleList([
            HypergraphConv(in_dim, in_dim // 2, dropout=dropout),
            HypergraphConv(in_dim // 2, out_dim, dropout=dropout)
        ])  # Move to device

    def forward(self, data, args):
        device = args.device
        X = copy.deepcopy(data.x)  # Original features
        X = normalize_l2(X)
        edge_index = data.edge_index.long()
        
        # Augmentation for edges based on overlap
        sel_mask = aug_node(data.overlap, args, device)


        for m in self.encoder:
            X = m(X,edge_index)
        Xve = X[edge_index[0]]
        Xe = scatter(Xve, edge_index[1], dim = 0,reduce = 'mean') #|E|*3
        Xe = normalize_l2(torch.sigmoid(Xe))

        sample = F.gumbel_softmax(Xe, hard=True)

        reserve = sample[:, 0].bool()
        mask = sample[:, 2].bool()


        reserve_sample = reserve[edge_index[1]]
        mask_sample = torch.logical_and(mask[edge_index[1]], sel_mask[edge_index[0]].to(device))
        final_sample = reserve_sample | mask_sample

        edge_index = edge_index[:, final_sample]
        return final_sample.float(), edge_index