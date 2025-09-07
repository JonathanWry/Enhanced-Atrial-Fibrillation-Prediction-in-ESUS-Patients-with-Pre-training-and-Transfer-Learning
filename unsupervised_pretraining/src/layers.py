#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
=====================================================================
AllSet Layers: PMA, MLP, HalfNLHconv (PyG/Hypergraph building blocks)
=====================================================================

File
----
layers.py  (paste this header at the top of the file)

Purpose
-------
Reusable layers for hypergraph/set-based models (e.g., AllSet variants):
  • PMA  : Pooling by Multihead Attention over bipartite incidence graphs
  • MLP  : Lightweight MLP with optional BN/LN & input normalization
  • HalfNLHconv : Half “Node→Edge/Edge→Node” block with optional attention
"""

import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional


# This part is for PMA.
# Modified from GATConv in pyg.
# Method for initialization
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class PMA(MessagePassing):
    """
        PMA part:
        Note that in original PMA, we need to compute the inner product of the seed and neighbor nodes.
        i.e. e_ij = a(Wh_i,Wh_j), where a should be the inner product, h_i is the seed and h_j are neighbor nodes.
        In GAT, a(x,y) = a^T[x||y]. We use the same logic.
    """
    _alpha: OptTensor

    def __init__(self, in_channels, hid_dim,
                 out_channels, num_layers, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, bias=False, **kwargs):
        super(PMA, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = 0.2
        self.aggr = 'add'
        # For neighbor nodes (source side, key)
        self.lin_K = Linear(in_channels, self.heads * self.hidden)
        # For neighbor nodes (source side, value)
        self.lin_V = Linear(in_channels, self.heads * self.hidden)
        self.att_r = Parameter(torch.Tensor(
            1, heads, self.hidden))  # Seed vector
        self.rFF = MLP(in_channels=self.heads * self.hidden,
                       hidden_channels=self.heads * self.hidden,
                       out_channels=out_channels,
                       num_layers=num_layers,
                       dropout=.0, Normalization='None', )
        self.ln0 = nn.LayerNorm(self.heads * self.hidden)
        self.ln1 = nn.LayerNorm(self.heads * self.hidden)
        self.register_parameter('bias', None)

        self._alpha = None
        self.small_constant=1e-8

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        nn.init.xavier_uniform_(self.att_r)


    def forward(self, x, edge_index: Adj,
                size: Size = None, return_attention_weights=None, edge_weight=None, cnum=None,max_index=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaNs or Infs detected in input x in PMA forward")
        self.cnum = cnum
        H, C = self.heads, self.hidden
        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaNs or Infs detected in input x in PMA forward")
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1))
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_K = self.lin_K(x).view(-1, H, C)
            if torch.isnan(x_K).any() or torch.isinf(x_K).any():
                print("NaNs or Infs detected in x_K")
            x_V = self.lin_V(x).view(-1, H, C)
            if torch.isnan(x_V).any() or torch.isinf(x_V).any():
                print("NaNs or Infs detected in x_V")
            alpha_r = (x_K * self.att_r).sum(dim=-1)
            if torch.isnan(alpha_r).any() or torch.isinf(alpha_r).any():
                print("NaNs or Infs detected in alpha_r")
        device = x.device
        edge_index = edge_index.to(device)
        x_V = x_V.to(device)
        alpha_r = alpha_r.to(device)
        edge_weight = edge_weight.to(device)

        # print(f"Before propagation: x_K shape: {x_K.shape}, x_V shape: {x_V.shape}, alpha_r shape: {alpha_r.shape}")
        if max_index is not None:
            out = self.propagate(edge_index.clone().to(device), x=x_V,
                                 alpha=alpha_r, aggr=self.aggr, edge_weight=edge_weight,
                                 max_index=max_index, size=size)
        else:
            out = self.propagate(edge_index.clone().to(device), x=x_V,
                                 alpha=alpha_r, aggr=self.aggr, edge_weight=edge_weight,max_index=None)
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("NaNs or Infs detected in output out in PMA propagation")
        # print(f"After propagation: out shape: {out.shape}, edge_index shape: {edge_index.shape}")
        alpha = self._alpha
        self._alpha = None

        #         Note that in the original code of GMT paper, they do not use additional W^O to combine heads.
        #         This is because O = softmax(QK^T)V and V = V_in*W^V. So W^O can be effectively taken care by W^V!!!
        out += self.att_r  # This is Seed + Multihead
        # concat heads then LayerNorm. Z (rhs of Eq(7)) in GMT paper.
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        # rFF and skip connection. Lhs of eq(7) in GMT paper.
        out = self.ln1(out + F.relu(self.rFF(out)))

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            # elif isinstance(edge_index, SparseTensor):
            #     return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j, alpha_j,
                index, ptr,
                size_j, edge_weight,max_index=None):
        #         ipdb.set_trace()
        alpha = alpha_j
        if torch.isnan(alpha).any() or torch.isinf(alpha).any():
            print("NaNs or Infs detected in output out in PMA alpha")
        alpha = F.leaky_relu(alpha, self.negative_slope)
        if torch.isnan(alpha).any() or torch.isinf(alpha).any():
            print("NaNs or Infs detected in output out in PMA alpha after leaky_relu")
        alpha = softmax(alpha, index, ptr, index.max()+1) #instead of index.max I think I should passed edge and node here!
        if torch.isnan(alpha).any() or torch.isinf(alpha).any():
            print("NaNs or Infs detected in output out in PMA alpha after softmax")
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if torch.isnan(alpha).any() or torch.isinf(alpha).any():
            print("NaNs or Infs detected in output out in PMA alpha after dropout")

        # print(f"Message function: alpha shape: {alpha.shape}, x_j shape: {x_j.shape}")
        
        if edge_weight is None:
            return x_j * alpha.unsqueeze(-1)  # Weighted by attention
        else:
            return x_j * alpha.unsqueeze(-1) * edge_weight.view(-1, 1, 1)

    def aggregate(self, inputs, index,
                  dim_size=None, aggr='sum',max_index=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        #         ipdb.set_trace()
        if aggr is None:
            raise ValueError("aggr was not passed!")
        if max_index is not None:
            dim_size = max_index
            aggregated = scatter(inputs, index, dim=self.node_dim, reduce=aggr,dim_size=dim_size)
        else:
            aggregated = scatter(inputs, index, dim=self.node_dim, reduce=aggr)
        # print(f"After aggregation: aggregated shape: {aggregated.shape}")
        if torch.isnan(aggregated).any() or torch.isinf(aggregated).any():
            print("NaNs or Infs detected in output out in PMA aggregated")
        return aggregated

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class HalfNLHconv(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout,
                 Normalization='bn',
                 InputNorm=False,
                 heads=1,
                 attention=True
                 ):
        super(HalfNLHconv, self).__init__()

        self.attention = attention
        self.dropout = dropout
        self.small_constant = 1e-8

        if self.attention:
            self.prop = PMA(in_dim, hid_dim, out_dim, num_layers, heads=heads)
        else:
            if num_layers > 0:
                self.f_enc = MLP(in_dim, hid_dim, hid_dim, num_layers, dropout, Normalization, InputNorm)
                self.f_dec = MLP(hid_dim, hid_dim, out_dim, num_layers, dropout, Normalization, InputNorm)
            else:
                self.f_enc = nn.Identity()
                self.f_dec = nn.Identity()

    #         self.bn = nn.BatchNorm1d(dec_hid_dim)
    #         self.dropout = dropout
    #         self.Prop = S2SProp()

    def reset_parameters(self):

        if self.attention:
            self.prop.reset_parameters()
        else:
            if not (self.f_enc.__class__.__name__ is 'Identity'):
                self.f_enc.reset_parameters()
            if not (self.f_dec.__class__.__name__ is 'Identity'):
                self.f_dec.reset_parameters()

    #         self.bn.reset_parameters()

    def forward(self, x, edge_index, norm, aggr='add', edge_weight=None, max_index=None):
        """
        input -> MLP -> Prop
        """
        device = x.device  # Get the device from the input tensor
        # print('edge_index',edge_index.device)
        # print('device',device)

        # Move other tensors to the same device as the input tensor
        edge_index = edge_index.to(device)
        if norm is not None:
            norm = norm.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        weight_tuple = None
        if self.attention:
            # print("Before PMA - x shape:", x.shape)
            self.prop.to(device)
            if max_index is not None:
                x, weight_tuple = self.prop(x, edge_index, edge_weight=edge_weight, return_attention_weights=True, max_index=max_index)
            else:
                x, weight_tuple = self.prop(x, edge_index, edge_weight=edge_weight, return_attention_weights=True)
            # print("After PMA - x shape:", x.shape)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("NaNs or Infs detected in input x before lin_K")
        else:
            x = F.relu(self.f_enc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.propagate(edge_index, x=x, norm=norm, aggr=aggr, edge_weight=edge_weight)
            x = F.relu(self.f_dec(x))
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("NaNs or Infs detected in input x before lin_K")

        return x, weight_tuple

    def message(self, x_j, norm, edge_weight):  # Add edge weight
        # Ensure tensors are on the same device
        zero_mask = (x_j == 0).float()  # Mask of zero features
        contribution = norm.view(-1, 1) * x_j
        contribution += zero_mask * self.small_constant
        device = x_j.device
        norm = norm.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        return contribution if edge_weight is None else contribution * edge_weight.view(-1, 1)

    def aggregate(self, inputs, index,
                  dim_size=None, aggr='add'):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        #         ipdb.set_trace()
        print("aggregate - inputs shape:", inputs.shape)
        print("aggregate - index shape:", index.shape)
        device = inputs.device
        index = index.to(device)
        if aggr is None:
            raise ValueError("aggr was not passed!")
        if aggr is None:
            raise ValueError("aggr was not passed!")
        adjusted_inputs = inputs + (inputs == 0).float() * self.small_constant
        return scatter(adjusted_inputs, index, dim=self.node_dim, reduce=aggr)
