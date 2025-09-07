#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
===============================================================================
Hypergraph Neural Network Models (SetGNN, ViewLearner, TuningMLP)
===============================================================================

File
----
models.py

Purpose
-------
Implementations used in our hypergraph experiments:

  • SetGNN
      Alternating V→E and E→V message passing via `HalfNLHconv`, with
      PairNorm-style normalization and Jumping Knowledge (concatenation of
      intermediate representations). Final MLP classifier over hyperedge
      embeddings.

  • ViewLearner
      Learns per-edge logits (drop/keep weights) conditioned on node/edge
      embeddings from an encoder (e.g., SetGNN), useful for contrastive
      view generation.

  • TuningMLP
      Small MLP utility module (ELU + sigmoid) for scalar tuning heads.

Core API
--------
1) SetGNN(args, data, norm=None)
   - forward(data, edge_weight=None)
       -> (edge_score, edge_feat, node_feat, aux_weights)
     * Uses `data.x`, `data.edge_index`, `data.norm`, `data.y`.
     * If `args.LearnFeat` is True, learns an embedding parameter initialized
       from `data.x`.
   - get_embedding(data, edge_weight=None)
       -> (edge_repr_concat, aux_weights)
     * Returns the concatenated Jumping-Knowledge representation before the
       classifier head.

2) ViewLearner(encoder, input_dim, viewer_hidden_dim=64)
   - forward(data, device)
       -> weight_logits aligned with `data.edge_index`
     * Expects `data.totedges` and `data.num_hyperedges` to separate true
       incidence edges from appended self-loops (self-loops are given large
       positive logits).

3) TuningMLP(input_size=32, hidden_size=128, output_size=1)
   - forward(x) -> sigmoid scalar(s)
   - get_embedding(x) -> hidden representation

Data Assumptions
----------------
- `data.edge_index` is the bipartite incidence (nodes↔hyperedges).
- `data.norm` contains per-incidence weights (optional but supported).
- `data.y` holds edge-level labels; SetGNN classifies hyperedges.
- `data.num_hyperedges` and `data.totedges` are required by ViewLearner to
  identify self-loop entries at the tail of `edge_index`.

Dependencies
------------
• PyTorch (`torch`, `torch.nn`, `torch.nn.functional`)
• torch_geometric (ops / base classes; custom layers are imported from `layers`)
• torch_scatter (`scatter`)
• NumPy

Notes
-----
- Message-passing layers (`HalfNLHconv`, `MLP`, `Linear`) are imported
  from the local `layers` module.
- The file imports `GCNConv` and `GATConv` but they are not used directly here.
- There is no `SimpleHypergraphModel` or KMeans-based pretraining in this file.
- No `pretrain_forward` is implemented in `SetGNN`.
"""


import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from layers import *

import math

from torch_scatter import scatter
from torch_geometric.utils import softmax

import numpy as np


class SetGNN(nn.Module):
    def __init__(self, args, data, norm=None):
        super(SetGNN, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """

        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = True
        self.LearnFeat = args.LearnFeat

        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()
        if self.LearnFeat:
            self.x = Parameter(data.x, requires_grad=True)


        if self.All_num_layers == 0:
            self.classifier = MLP(in_channels=args.num_features,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.num_labels,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
        else:
            self.V2EConvs.append(HalfNLHconv(in_dim=args.feature_dim,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA))
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            for i in range(self.All_num_layers):
                self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
                self.V2EConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA))
                if i < self.All_num_layers-1:
                    self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.classifier = MLP(
                                  # in_channels=args.MLP_hidden,
                                  in_channels=args.MLP_hidden * (args.All_num_layers + 1),
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.num_labels,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data, edge_weight=None):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnFeat:
            x = self.x

        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)

        vec = []
        x = F.dropout(x, p=0.2, training=self.training)  # Input dropout

        scale = 1
        eps = 1e-5
        for i, _ in enumerate(self.E2VConvs):
            x, weight_tuple = self.V2EConvs[i](x, edge_index, norm, self.aggr, edge_weight=edge_weight)
            # PairNorm
            x = x - x.mean(dim=0, keepdim=True)
            x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
            # Jumping Knowledge
            vec.append(x)
            x = self.bnV2Es[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x, weight_tuple = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr, edge_weight=edge_weight)
            # PairNorm
            x = x - x.mean(dim=0, keepdim=True)
            x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
            node_feat = x
            x = self.bnE2Vs[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x, weight_tuple = self.V2EConvs[-1](x, edge_index, norm, self.aggr, edge_weight=edge_weight)
        # PairNorm
        x = x - x.mean(dim=0, keepdim=True)
        x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
        edge_feat = x
        # Jumping Knowledge
        vec.append(x)

        x = torch.cat(vec, dim=1)
        x = x[:data.y.shape[0], :]
        edge_score = self.classifier(x)

        return edge_score, edge_feat, node_feat, weight_tuple


    def get_embedding(self, data, edge_weight=None):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
        #             The data should contain the follows
        #             data.x: node features
        #             data.V2Eedge_index:  edge list (of size (2,|E|)) where
        #             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges

        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnFeat:
            x = self.x

        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)

        vec = []
        x = F.dropout(x, p=0.2, training=self.training)  # Input dropout

        scale = 1
        eps = 1e-5
        for i, _ in enumerate(self.E2VConvs):
            x, weight_tuple = self.V2EConvs[i](x, edge_index, norm, self.aggr, edge_weight=edge_weight)
            # PairNorm
            x = x - x.mean(dim=0, keepdim=True)
            x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
            # Jumping Knowledge
            vec.append(x)
            x = self.bnV2Es[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x, weight_tuple = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr, edge_weight=edge_weight)
            # PairNorm
            x = x - x.mean(dim=0, keepdim=True)
            x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
            node_feat = x
            x = self.bnE2Vs[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x, weight_tuple = self.V2EConvs[-1](x, edge_index, norm, self.aggr, edge_weight=edge_weight)
        # PairNorm
        x = x - x.mean(dim=0, keepdim=True)
        x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
        edge_feat = x
        # Jumping Knowledge
        vec.append(x)

        x = torch.cat(vec, dim=1)
        x = x[:data.y.shape[0], :]
        # edge_score = self.classifier(x)

        return x, weight_tuple



class ViewLearner(torch.nn.Module):
    """
    Learns edge-wise logits for contrastive view generation.

    Given an encoder (e.g., SetGNN) that produces node and hyperedge embeddings,
    ViewLearner conditions on [node, hyperedge] pairs to predict dropout/retention
    weights for each incidence edge. Self-loops are handled specially and assigned
    large positive logits.

    Args:
        encoder (nn.Module): Base encoder returning (edge_score, edge_feat, node_feat, aux).
        input_dim (int): Dimensionality of encoder embeddings (per node/hyperedge).
        viewer_hidden_dim (int, optional): Hidden dimension of the MLP predictor. Default=64.

    Methods:
        forward(data, device):
            Runs the encoder, gathers per-(node, hyperedge) embeddings, and outputs
            logits aligned with `data.edge_index` (first true hyperedges, then self-loops).
    """

    def __init__(self, encoder, input_dim, viewer_hidden_dim=64):
        super(ViewLearner, self).__init__()

        self.encoder = encoder
        self.input_dim = input_dim

        self.mlp_edge_model = nn.Sequential(
            Linear(self.input_dim * 2, viewer_hidden_dim),
            nn.ReLU(),
            Linear(viewer_hidden_dim, 1)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data, device):
        # Encode once to obtain node and hyperedge embeddings from the current graph
        _, edge_feat, node_feat, _ = self.encoder(data.clone())
        # Split true hyperedges (first) vs. self-loops (last) using counts carried in data
        totedges = data.totedges
        num_hyperedges = data.num_hyperedges 
        num_self_loop = totedges - num_hyperedges
        edge_index = data.edge_index.clone()
        # Convert to Python int for slicing (handles tensor inputs)
        num_self_loop_clone = int(num_self_loop)
        node, edge = edge_index[:, :-num_self_loop_clone][0], edge_index[:, :-num_self_loop_clone][1]
        # Gather per-(node, hyperedge) embeddings
        emb_node = node_feat[node]
        emb_edge = edge_feat[edge]
        # Concatenate [node | hyperedge] embeddings and predict an edge logit
        total_emb = torch.cat([emb_node, emb_edge], 1)
        edge_weight = self.mlp_edge_model(total_emb)
        # Reassemble logits aligned with original edge_index order (true edges + self-loops)
        self_loop_weight = np.ones(shape=(num_self_loop_clone, 1)) * 10.0
        self_loop_weight = torch.FloatTensor(self_loop_weight).to(device)
        weight_logits = torch.cat([edge_weight, self_loop_weight], 0)

        return weight_logits



class TuningMLP(nn.Module):
    """
      A lightweight multi-layer perceptron (MLP) used for tuning/scoring tasks.

      Architecture:
          - Linear(input_size → hidden_size) + ELU
          - Linear(hidden_size → output_size)
          - Sigmoid activation on output

      Methods:
          forward(x):
              Returns sigmoid-scaled scores of shape (..., output_size).

          get_embedding(x):
              Returns hidden representation after the first layer (before output).
      """
    def __init__(self, input_size=32, hidden_size=128, output_size=1):
        super(TuningMLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.ELU())
        # self.layer2 = nn.Sequential(nn.Linear(hidden_size, input_size),
        #                             nn.ELU())
        # self.layer3 = nn.Linear(input_size, output_size)
        self.layer2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = torch.sigmoid(x)
        return x
    
    def get_embedding(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        return x 


