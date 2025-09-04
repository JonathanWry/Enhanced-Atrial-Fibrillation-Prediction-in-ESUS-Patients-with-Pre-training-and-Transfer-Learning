"""
===============================================================================
Hypergraph Neural Network Models (SetGNN, SimpleHypergraphModel, ViewLearner)
===============================================================================

File
----
models.py   (paste this header at the top of the file)

Purpose
-------
Collection of hypergraph-based GNN architectures and contrastive losses used in
our paper. Includes:

  • SetGNN
      Alternating V→E and E→V message passing with optional Jumping Knowledge,
      normalization, and hyperedge/node-level contrastive pretraining losses.

  • SimpleHypergraphModel
      Minimal 2-layer HypergraphConv encoder + MLP classifier for hyperedge
      prediction. Useful as a lightweight baseline.

  • ViewLearner
      Learns edge dropout/retention logits conditioned on encoder outputs for
      contrastive view generation.

Core API
--------
1) SetGNN(args, data, norm=None)
   - forward(data, …) -> (edge_score, edge_feat, node_feat, aux)
   - pretrain_forward(…) -> Dict[str, Tensor] of task losses
   - Includes node-level, edge-level, membership-level contrastive objectives.

2) SimpleHypergraphModel(in_dim, hidden_dim, out_dim, num_labels)
   - forward(data, …) -> (edge_scores, edge_embs, node_embs, aux)
   - Provides loss functions for node/edge contrast and membership-level tasks.

3) ViewLearner(encoder, input_dim, viewer_hidden_dim=64)
   - forward(data, device) -> edge logits aligned with edge_index
   - Used to parameterize edge masking for augmented views.

Dependencies
------------
• PyTorch, torch_geometric (MessagePassing, HypergraphConv, scatter ops)
• scikit-learn (KMeans for clustering in pretraining)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from torch.nn import Linear, Parameter
from sklearn.cluster import KMeans
from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
import math
from torch_scatter import scatter, scatter_mean
from torch_geometric.utils import softmax
import numpy as np
from torch_geometric.nn import HypergraphConv

from layers import *


class SetGNN(nn.Module):
    """
    SetGNN: Alternating V→E and E→V message passing over a hypergraph incidence.
    Produces (hyper)edge and node embeddings + a classifier for hyperedge scores.

    Expects:
      - data.x: [|V|, d_in] node features
      - data.edge_index: [2, |I|] incidence (row: nodes, col: hyperedges; 0-based IDs)
      - data.norm: optional incidence weights aligned with edge_index

    Args of interest:
      - All_num_layers: # of E↔V alternating blocks (≥1 recommended)
      - MLP_hidden / _num_layers: width/depth of HalfNLHconv inner MLPs
      - Classifier_hidden / _num_layers: hyperedge classifier head
      - aggregate: 'mean' or 'sum' for scatter aggregation
    """


    def __init__(self, args, data, norm=None):
        super(SetGNN, self).__init__()

        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = True
        self.LearnFeat = args.LearnFeat
        # Stacks of alternating message passing:
        # V2EConvs: aggregate node -> hyperedge
        # E2VConvs: aggregate hyperedge -> node
        # BatchNorms are paired with each stage.

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
                if i < self.All_num_layers - 1:
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
            self.disc = nn.Bilinear(args.MLP_hidden, args.MLP_hidden, 1)

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

    def forward(self, data, args=None,edge_mask=None, edge_weight=None, edge_size=None,node_size=None):
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
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaNs or Infs detected in input x before lin_K")
        device = data.x.device
        # print('model->device',device)
        # print('model-edge_index',edge_index.device)
        if self.LearnFeat:
            x = self.x
        #
        data = data.to(device)
        edge_index = edge_index.to(device)
        if norm is not None:
            norm = norm.to(device)
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)

        vec = []
        x = F.dropout(x, p=0.2, training=self.training)  # Input dropout

        scale = 1
        eps = 1e-5
        for i, _ in enumerate(self.E2VConvs):
            if edge_size is not None:
                x, weight_tuple = self.V2EConvs[i](x, edge_index, norm, self.aggr,
                                                   edge_weight=edge_weight,max_index=edge_size)  # 这里传入限制的边和node
            else:
                x, weight_tuple = self.V2EConvs[i](x, edge_index, norm, self.aggr,
                                                   edge_weight=edge_weight)
            x = x - x.mean(dim=0, keepdim=True)
            x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
            # Jumping Knowledge
            vec.append(x)
            x = self.bnV2Es[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if node_size is not None:
                x, weight_tuple = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr, edge_weight=edge_weight, max_index=node_size)
            else:
                x, weight_tuple = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr, edge_weight=edge_weight)
            # PairNorm
            x = x - x.mean(dim=0, keepdim=True)
            x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
            node_feat = x
            x = self.bnE2Vs[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if edge_size is not None:
            x, weight_tuple = self.V2EConvs[i+1](x, edge_index, norm, self.aggr,
                                               edge_weight=edge_weight, max_index=edge_size)  # 这里传入限制的边和node
        else:
            x, weight_tuple = self.V2EConvs[i+1](x, edge_index, norm, self.aggr,
                                               edge_weight=edge_weight)
        # PairNorm
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaNs or Infs detected in input x before lin_K")
        x = x - x.mean(dim=0, keepdim=True)
        x = scale * x / (eps + x.pow(2).sum(-1).mean()).sqrt()
        edge_feat = x
        # Jumping Knowledge
        vec.append(x)

        x = torch.cat(vec, dim=1)
        x = x[:data.y.shape[0], :]
        edge_score = self.classifier(x)

        return edge_score, edge_feat, node_feat, weight_tuple

    def f(self, x, tau):
        if tau == 0:
            raise ValueError("Tau cannot be zero in the exponential function.")
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input 'x' contains NaN or infinite values.")
        return torch.exp(x / tau)

    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        eps = 1e-8
        z1_norm = torch.norm(z1, dim=1, p=2).clamp(min=eps)
        z2_norm = torch.norm(z2, dim=1, p=2).clamp(min=eps)
        z1 = z1 / z1_norm.unsqueeze(1)
        z2 = z2 / z2_norm.unsqueeze(1)
        if z1.shape[1] != z2.shape[1]:
            raise ValueError("Mismatched shapes between z1 and z2 in cosine_similarity.")
        similarity = torch.mm(z1, z2.t())
        similarity = torch.clamp(similarity, min=-1.0, max=1.0)
        if torch.isnan(similarity).any() or torch.isinf(similarity).any():
            raise ValueError("NaN or Inf values detected in similarity matrix.")
        return similarity

    def disc_similarity(self, z1: Tensor, z2: Tensor):
        return torch.sigmoid(self.disc(z1, z2)).squeeze()

    def __loss(self, z1: Tensor, z2: Tensor, tau: float, batch_size: Optional[int],
               num_negs: Optional[int], mean: bool):
        """
        Symmetric contrastive loss wrapper.

        Computes an InfoNCE-style loss in both directions (z1→z2 and z2→z1),
        using either full in-batch negatives (`__semi_loss`) or a batched
        variant (`__semi_loss_batch`) depending on arguments.

        Args:
            z1 (Tensor): First embedding set (N, D).
            z2 (Tensor): Second embedding set (N, D).
            tau (float): Temperature (>0) for similarity scaling.
            batch_size (Optional[int]): If set, uses batched negatives.
            num_negs (Optional[int]): If set, samples K negatives per anchor.
                Note: when `num_negs` is not None, the full-matrix path is used.
            mean (bool): If True, mean-reduce across samples; else sum.

        Returns:
            Tensor: Scalar loss (mean/sum over samples).
        """
        if batch_size is None or num_negs is not None:
            l1 = self.__semi_loss(z1, z2, tau, num_negs)
            l2 = self.__semi_loss(z2, z1, tau, num_negs)
        else:
            l1 = self.__semi_loss_batch(z1, z2, tau, batch_size)
            l2 = self.__semi_loss_batch(z2, z1, tau, batch_size)

        loss = (l1 + l2) * 0.5
        loss = loss.mean() if mean else loss.sum()
        return loss

    def __semi_loss(self, h1: Tensor, h2: Tensor, tau: float, num_negs: Optional[int]):
        """
        Compute an InfoNCE-style contrastive loss between two embedding sets.

        Args:
            h1 (Tensor): Anchor embeddings of shape (N, D).
            h2 (Tensor): Positive/negative pool embeddings of shape (N, D).
            tau (float): Temperature for softmax scaling (> 0).
            num_negs (Optional[int]): If None, use full in-batch negatives (full matrix).
                If an integer K is provided, sample K negative permutations of h2.

        Returns:
            Tensor: Per-sample loss vector of shape (N,).

        Notes:
            - Uses cosine similarity, exponentiated and scaled by tau.
            - Full negatives path forms a (N×N) similarity matrix; diagonal is positive.
            - Sampled negatives path draws K permuted copies of h2 for each anchor.
            - Raises ValueError when NaN/Inf is detected in inputs.
        """
        if torch.isnan(h1).any() or torch.isnan(h2).any():
            raise ValueError("NaN values detected in h1 or h2 before normalization.")
        if torch.isinf(h1).any() or torch.isinf(h2).any():
            raise ValueError("Inf values detected in h1 or h2 before normalization.")
        if num_negs is None:
            between_sim = self.f(self.cosine_similarity(h1, h2), tau)
            return -torch.log(between_sim.diag() / between_sim.sum(1))
        else:
            pos_sim = self.f(F.cosine_similarity(h1, h2), tau)
            negs = []
            for _ in range(num_negs):
                negs.append(h2[torch.randperm(h2.size(0))])
            negs = torch.stack(negs, dim=-1)
            neg_sim = self.f(F.cosine_similarity(h1.unsqueeze(-1).tile(num_negs), negs), tau)
            return -torch.log(pos_sim / (pos_sim + neg_sim.sum(1)))

    def __semi_loss_batch(self, h1: Tensor, h2: Tensor, tau: float, batch_size: int):
        """
        Batched variant of __semi_loss with full in-batch negatives.

        Args:
            h1 (Tensor): Anchor embeddings of shape (N, D).
            h2 (Tensor): Positive/negative pool embeddings of shape (N, D).
            tau (float): Temperature for softmax scaling (> 0).
            batch_size (int): Mini-batch size for anchors (rows of h1).

        Returns:
            Tensor: Concatenated per-sample loss vector of shape (N,).

        Notes:
            - For each batch of anchors, computes similarity to all rows of h2.
            - Positive for row i is column i within the current batch window.
            - Performs numerical checks for zero vectors and NaN/Inf values.
        """
        if torch.isnan(h1).any() or torch.isnan(h2).any():
            raise ValueError("NaN values detected in h1 or h2 before normalization.")
        if torch.isinf(h1).any() or torch.isinf(h2).any():
            raise ValueError("Inf values detected in h1 or h2 before normalization.")
        device = h1.device
        num_samples = h1.size(0)
        num_batches = (num_samples - 1) // batch_size + 1
        indices = torch.arange(0, num_samples, device=device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]
            if torch.any(torch.norm(h1[mask], dim=1) == 0) or torch.any(torch.norm(h2, dim=1) == 0):
                raise ValueError("Zero vectors detected in h1 or h2 before cosine similarity.")

            between_sim = self.f(self.cosine_similarity(h1[mask], h2), tau)

            loss = -torch.log(between_sim[:, i * batch_size: (i + 1) * batch_size].diag() / between_sim.sum(1))
            losses.append(loss)
        return torch.cat(losses)

    def f(self, x, tau):
        """
       Temperature-scaled exponential used in contrastive scoring.

       Args:
           x (Tensor): Similarity scores (any shape), typically cosine similarities.
           tau (float): Temperature (> 0). Smaller tau sharpens the distribution.

       Returns:
           Tensor: exp(x / tau), same shape as x.

       Raises:
           ValueError: If tau == 0 or x contains NaN/Inf.
       """
        if tau == 0:
            raise ValueError("Tau cannot be zero in the exponential function.")
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input 'x' contains NaN or infinite values.")
        return torch.exp(x / tau)

    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        eps = 1e-8
        z1_norm = torch.norm(z1, dim=1, p=2).clamp(min=eps)
        z2_norm = torch.norm(z2, dim=1, p=2).clamp(min=eps)
        z1 = z1 / z1_norm.unsqueeze(1)
        z2 = z2 / z2_norm.unsqueeze(1)
        if z1.shape[1] != z2.shape[1]:
            raise ValueError("Mismatched shapes between z1 and z2 in cosine_similarity.")
        similarity = torch.mm(z1, z2.t())
        similarity = torch.clamp(similarity, min=-1.0, max=1.0)
        if torch.isnan(similarity).any() or torch.isinf(similarity).any():
            raise ValueError("NaN or Inf values detected in similarity matrix.")
        return similarity

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def cal_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        Row-wise InfoNCE numerator/denominator with temperature (tau=0.5).

        Builds a contrastive objective using:
          - refl_sim: similarities within z1 (self-similarity matrix)
          - between_sim: similarities between z1 (rows) and z2 (cols)
        The positive for row i is column i; all other columns are negatives.

        Args:
            z1 (Tensor): Embeddings A of shape (N, D).
            z2 (Tensor): Embeddings B of shape (N, D).

        Returns:
            Tensor: Per-sample loss vector (N,).
        """
        self.tau = 0.5
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        eps = 1e-8
        denominator = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() + eps
        safe_ratio = torch.clamp(between_sim.diag() / denominator, min=1e-8)

        return -torch.log(safe_ratio)

    def loss_hyperedge_ada_maxmargin(self, Z1, Z2):
        """
        Symmetric hyperedge contrast with adaptive margin (via tau inside cal_loss).

        Applies `cal_loss` in both directions on transposed inputs so that
        rows correspond to comparable items (hyperedges), then averages.

        Args:
            Z1 (Tensor): Embeddings from view 1, shape (D, N) or (N, D).
            Z2 (Tensor): Embeddings from view 2, shape (D, N) or (N, D).

        Returns:
            Tensor: Scalar loss (mean over samples).
        """
        h1 = Z1.T
        h2 = Z2.T
        l1 = self.cal_loss(h1, h2)
        l2 = self.cal_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret

    def node_level_loss(self, n1: Tensor, n2: Tensor, node_tau: float,
                        batch_size: Optional[int] = None, num_negs: Optional[int] = None,
                        mean: bool = True):
        """
        Node-level contrastive loss between two augmented views.

        Args:
            n1 (Tensor): Node embeddings from view 1 (N, D).
            n2 (Tensor): Node embeddings from view 2 (N, D).
            node_tau (float): Temperature for node-level loss.
            batch_size (Optional[int]): If set, use batched negatives.
            num_negs (Optional[int]): If set, sample K negatives per anchor.
            mean (bool): Mean- or sum-reduction over samples.

        Returns:
            Tensor: Scalar node-level loss.
        """
        loss = self.__loss(n1, n2, node_tau, batch_size, num_negs, mean)
        return loss

    def group_level_loss(self, e1: Tensor, e2: Tensor, edge_tau: float,
                         batch_size: Optional[int] = None, num_negs: Optional[int] = None,
                         mean: bool = True):
        """
        Hyperedge-level (group-level) contrastive loss between two views.

        Args:
            e1 (Tensor): Hyperedge embeddings from view 1 (|E|, D).
            e2 (Tensor): Hyperedge embeddings from view 2 (|E|, D).
            edge_tau (float): Temperature for edge-level loss.
            batch_size (Optional[int]): If set, use batched negatives.
            num_negs (Optional[int]): If set, sample K negatives per anchor.
            mean (bool): Mean- or sum-reduction over samples.

        Returns:
            Tensor: Scalar hyperedge-level loss.
        """
        loss = self.__loss(e1, e2, edge_tau, batch_size, num_negs, mean)
        return loss

    def membership_level_loss_with_clusters(self, n: Tensor, e: Tensor, hyperedge_index: Tensor, tau: float,
                                            cluster_assignments_n: Tensor, cluster_assignments_e: Tensor,
                                            batch_size: Optional[int] = None, mean: bool = True):
        """
        Modified membership level loss function to use precomputed cluster assignments and a subset of positive and negative samples.

        Args:
            n (Tensor): Node embeddings.
            e (Tensor): Edge embeddings.
            hyperedge_index (Tensor): Hyperedge index tensor.
            tau (float): Temperature parameter.
            cluster_assignments_n (Tensor): Precomputed cluster assignments for node embeddings.
            cluster_assignments_e (Tensor): Precomputed cluster assignments for edge embeddings.
            batch_size (Optional[int]): Batch size for processing large datasets.
            mean (bool): If True, average the loss; otherwise, sum it.
            num_pos_samples (Optional[int]): Number of positive samples to use. If None, use all positives.
            num_neg_samples (Optional[int]): Number of negative samples to use. If None, use all negatives.

        Returns:
            Tensor: Computed loss.
        """
        # Permute the edge and node embeddings for negative samples
        e_perm = e[torch.randperm(e.size(0))]
        n_perm = n[torch.randperm(n.size(0))]

        # Determine the number of available samples
        num_available_pos = hyperedge_index.shape[1]
        num_available_neg = min(e_perm.size(0), n_perm.size(0))

        if batch_size is None:
            # Without batching
            pos = self.f(self.disc_similarity(n[hyperedge_index[0]], e[hyperedge_index[1]]), tau)
            neg_n = self.f(self.disc_similarity(n[hyperedge_index[0]], e_perm[hyperedge_index[1] % e_perm.size(0)]),
                           tau)
            neg_e = self.f(self.disc_similarity(n_perm[hyperedge_index[0] % n_perm.size(0)], e[hyperedge_index[1]]),
                           tau)

            # Use cluster-based weights as pseudo-labels
            cluster_weights = (
                    cluster_assignments_n[hyperedge_index[0]] == cluster_assignments_e[hyperedge_index[1]]).float()
            loss_n = -torch.log(pos / (pos + neg_n)) * cluster_weights
            loss_e = -torch.log(pos / (pos + neg_e)) * cluster_weights
        else:
            # With batching
            num_samples = hyperedge_index.shape[1]
            num_batches = (num_samples - 1) // batch_size + 1
            indices = torch.arange(0, num_samples, device=n.device)
            aggr_pos = []
            aggr_neg_n = []
            aggr_neg_e = []
            aggr_weights = []
            for i in range(num_batches):
                mask = indices[i * batch_size: (i + 1) * batch_size]
                if len(mask) > 0:
                    pos = self.f(self.disc_similarity(n[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]),
                                 tau)
                    if pos.shape != ():
                        neg_n = self.f(
                            self.disc_similarity(n[hyperedge_index[:, mask][0]],
                                                 e_perm[hyperedge_index[:, mask][1] % e_perm.size(0)]),
                            tau)
                        neg_e = self.f(
                            self.disc_similarity(n_perm[hyperedge_index[:, mask][0] % n_perm.size(0)],
                                                 e[hyperedge_index[:, mask][1]]),
                            tau)

                        # Append results only if mask > 0 and pos is not 0-dimensional
                        aggr_pos.append(pos)
                        aggr_neg_n.append(neg_n)
                        aggr_neg_e.append(neg_e)

                        # Collect cluster weights
                        aggr_weights.append((cluster_assignments_n[hyperedge_index[:, mask][0]] ==
                                             cluster_assignments_e[hyperedge_index[:, mask][1]]).float())

            aggr_pos = torch.concat(aggr_pos)
            aggr_neg_n = torch.concat(aggr_neg_n)
            aggr_neg_e = torch.concat(aggr_neg_e)
            aggr_weights = torch.concat(aggr_weights)

            # Compute loss with cluster weighting
            loss_n = -torch.log(aggr_pos / (aggr_pos + aggr_neg_n))
            loss_e = -torch.log(aggr_pos / (aggr_pos + aggr_neg_e))

            # Apply cluster-based weighting (pseudo-labels)
            # Compute loss with cluster-based weights
            loss_n = -torch.log(aggr_pos / (aggr_pos + aggr_neg_n)) * aggr_weights
            loss_e = -torch.log(aggr_pos / (aggr_pos + aggr_neg_e)) * aggr_weights

        loss_n = loss_n[~torch.isnan(loss_n)]
        loss_e = loss_e[~torch.isnan(loss_e)]
        loss = loss_n + loss_e
        loss = loss.mean() if mean else loss.sum()

        return loss

    def membership_level_loss(self, n: Tensor, e: Tensor, hyperedge_index: Tensor, tau: float,
                              batch_size: Optional[int] = None, mean: bool = True,
                              num_samples: Optional[int] = None):
        """
        Calculate the membership-level loss.

        Parameters:
        - n: Tensor of node features.
        - e: Tensor of edge (hyperedge) features.
        - hyperedge_index: Tensor of indices representing hyperedges.
        - tau: Temperature parameter for the similarity function.
        - batch_size: Optional batch size for batch processing.
        - mean: Boolean indicating whether to average the loss.
        - num_samples: Number of samples to limit the processing to.

        Returns:
        - The membership-level loss.
        """

        # If num_samples is provided, limit the number of samples in hyperedge_index
        if num_samples is not None:
            num_samples = min(hyperedge_index.shape[1], num_samples)
            perm = torch.randperm(hyperedge_index.shape[1])[:num_samples]
            hyperedge_index = hyperedge_index[:, perm]

        # Permute edges and nodes for creating negative samples
        e_perm = e[torch.randperm(e.size(0))]
        n_perm = n[torch.randperm(n.size(0))]

        if batch_size is None:
            # Process without batching
            pos = self.f(self.disc_similarity(n[hyperedge_index[0]], e[hyperedge_index[1]]), tau)
            neg_n = self.f(self.disc_similarity(n[hyperedge_index[0]], e_perm[hyperedge_index[1]]), tau)
            neg_e = self.f(self.disc_similarity(n_perm[hyperedge_index[0]], e[hyperedge_index[1]]), tau)

            # Calculate losses for nodes and edges
            loss_n = -torch.log(pos / (pos + neg_n))
            loss_e = -torch.log(pos / (pos + neg_e))
        else:
            # Process with batching
            num_samples = hyperedge_index.shape[1]
            num_batches = (num_samples - 1) // batch_size + 1
            indices = torch.arange(0, num_samples, device=n.device)

            aggr_pos = []
            aggr_neg_n = []
            aggr_neg_e = []
            for i in range(num_batches):
                mask = indices[i * batch_size: (i + 1) * batch_size]
                if len(mask) > 0:
                    pos = self.f(self.disc_similarity(n[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]),
                                 tau)
                    if pos.shape != ():
                        neg_n = self.f(
                            self.disc_similarity(n[hyperedge_index[:, mask][0]], e_perm[hyperedge_index[:, mask][1]]),
                            tau)
                        neg_e = self.f(
                            self.disc_similarity(n_perm[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]),
                            tau)

                        # Append results only if mask > 0 and pos is not 0-dimensional
                        aggr_pos.append(pos)
                        aggr_neg_n.append(neg_n)
                        aggr_neg_e.append(neg_e)

            # Concatenate aggregated results
            aggr_pos = torch.cat(aggr_pos)
            aggr_neg_n = torch.cat(aggr_neg_n)
            aggr_neg_e = torch.cat(aggr_neg_e)

            # Calculate losses for nodes and edges
            loss_n = -torch.log(aggr_pos / (aggr_pos + aggr_neg_n))
            loss_e = -torch.log(aggr_pos / (aggr_pos + aggr_neg_e))

        # Remove NaN values from the losses
        loss_n = loss_n[~torch.isnan(loss_n)]
        loss_e = loss_e[~torch.isnan(loss_e)]

        # Aggregate the loss values
        loss = loss_n + loss_e
        return loss.mean() if mean else loss.sum()



    def compute_cluster_assignments(self, embeddings: Tensor, num_clusters: int) -> Tensor:
        """
        Compute cluster assignments for the given embeddings using KMeans.
        Args:
            embeddings (Tensor): The embeddings to cluster.
            num_clusters (int): Number of clusters.
        Returns:
            Tensor: Cluster assignments.
        """
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        clusters = kmeans.fit_predict(embeddings.detach().cpu().numpy())
        return torch.tensor(clusters, dtype=torch.long, device=embeddings.device)

    def pretrain_forward(self, n1, n2, e1, e2, edge_mask, edge_mask1, edge_mask2, masked_index1, masked_index2,
                         args, view_gen1, view_gen2, data, num_nodes, num_edges, num_negs, model, supervised=0):
        """
        Aggregate all pretraining losses by task name (genSim/node/graph/membership/supervised).
        Returns a dict mapping task->loss for weighted/pareto schedulers.
        """

        res = {}
        if 'genSim' in args.tasks:
            res['genSim'], _, _, _, _ = self.chgnn(data=data, args=args, view_gen1=view_gen1, view_gen2=view_gen2,
                                                   model=model, num_nodes=num_nodes, num_edges=num_edges)

        if 'node' in args.tasks:
            res['node'] = self.node_level_loss(n1, n2, args.pretrain_tau_n, batch_size=args.pretrain_ng_batch_size,
                                               num_negs=num_negs)

        if 'graph' in args.tasks:
            res['graph'] = self.group_level_loss(e1[edge_mask], e2[edge_mask], args.pretrain_tau_g,
                                                 batch_size=args.pretrain_ng_batch_size,
                                                 num_negs=num_negs)

        if 'membership' in args.tasks:
            loss_m1 = self.membership_level_loss(
                n=n1,
                e=e2[edge_mask2],
                hyperedge_index=masked_index2,
                tau=args.pretrain_tau_m,
                batch_size=args.pretrain_m_batch_size,
                mean=True
            )

            loss_m2 = self.membership_level_loss(
                n=n2,
                e=e1[edge_mask1],
                hyperedge_index=masked_index1,
                tau=args.pretrain_tau_m,
                batch_size=args.pretrain_m_batch_size,
                mean=True
            )
            res['membership'] = (loss_m1 + loss_m2) * 0.5
        if 'supervised' in args.tasks:
            res['supervised'] = supervised
        return res;

    def concatenate_tensors(self, tensor_list):
        """
        Concatenate tensors from a list while handling zero-dimensional tensors.
        """
        # Initialize an empty list for concatenation
        tensors_to_concatenate = []

        # Iterate through the tensor list
        for tensor in tensor_list:
            # If the tensor is not zero-dimensional, add it to the list for concatenation
            if tensor.dim() > 0:
                tensors_to_concatenate.append(tensor)
            else:
                # Skip zero-dimensional tensors or replace them with an empty tensor of appropriate shape
                # Alternatively, you can choose to log a warning or handle the zero-dimensional tensor in another way
                pass

        # If there are tensors to concatenate, perform the concatenation
        if tensors_to_concatenate:
            concatenated_tensor = torch.concat(tensors_to_concatenate)
            return concatenated_tensor

        # If no tensors to concatenate, return an empty tensor of appropriate type and device
        else:
            # Return an empty tensor with the same dtype and device as the first tensor in the original list
            if tensor_list:
                dtype = tensor_list[0].dtype
                device = tensor_list[0].device
                return torch.tensor([], dtype=dtype, device=device)
            else:
                return torch.tensor([])


class SimpleHypergraphModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_labels):
        super(SimpleHypergraphModel, self).__init__()
        # Define two HypergraphConv layers
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, out_channels)

        # Simple classifier for hyperedge score prediction
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_labels)  # Predicting score based on num_labels
        )
        self.disc = nn.Bilinear(hidden_channels, hidden_channels, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters for all layers in the model.
        This function reinitializes the weights and biases for each layer.
        """
        self.conv1.reset_parameters()  # Reset GCNConv layer 1
        self.conv2.reset_parameters()  # Reset GCNConv layer 2
        for layer in self.classifier:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, data, args=None, edge_mask=None,edge_weight=None,edge_size=None):
        x, edge_index = data.x, data.edge_index

        # Node to edge propagation
        x = F.relu(self.conv1(x, edge_index))
        node_embeddings = self.conv2(x, edge_index)
        hyperedge_embeddings = self.aggregate_embeddings(node_embeddings, edge_index, edge_size=edge_size)
        hyperedge_scores = self.classifier(hyperedge_embeddings)
        weight_tuple = (torch.tensor(0.0), torch.tensor(0.0))
        return hyperedge_scores,hyperedge_embeddings,node_embeddings,weight_tuple

    def aggregate_embeddings(self, node_embeddings, edge_index, edge_size=None, aggr='mean'):
        """
        Aggregates node embeddings into edge embeddings. If edge_size is provided, restrict the size of the output.

        Parameters:
        - node_embeddings: Tensor of node embeddings
        - edge_index: The node-to-edge index tensor
        - edge_size: Optional. Restricts the output size to edge_size.
        - aggr: Aggregation method (e.g., 'mean', 'sum', etc.)

        Returns:
        - Aggregated edge embeddings
        """
        node_idx = edge_index[0]
        edge_idx = edge_index[1]

        if edge_size is not None:
            dim_size = edge_size
            # Restrict the output size using edge_size in scatter operation
            edge_embeddings = scatter(node_embeddings[node_idx], edge_idx, dim=0, reduce=aggr, dim_size=dim_size)
        else:
            # No restriction on output size
            edge_embeddings = scatter(node_embeddings[node_idx], edge_idx, dim=0, reduce=aggr)

        return edge_embeddings

    def __loss(self, z1: Tensor, z2: Tensor, tau: float, batch_size: Optional[int],
               num_negs: Optional[int], mean: bool):
        if batch_size is None or num_negs is not None:
            l1 = self.__semi_loss(z1, z2, tau, num_negs)
            l2 = self.__semi_loss(z2, z1, tau, num_negs)
        else:
            l1 = self.__semi_loss_batch(z1, z2, tau, batch_size)
            l2 = self.__semi_loss_batch(z2, z1, tau, batch_size)

        loss = (l1 + l2) * 0.5
        loss = loss.mean() if mean else loss.sum()
        return loss

    def __semi_loss(self, h1: Tensor, h2: Tensor, tau: float, num_negs: Optional[int]):
        if torch.isnan(h1).any() or torch.isnan(h2).any():
            raise ValueError("NaN values detected in h1 or h2 before normalization.")
        if torch.isinf(h1).any() or torch.isinf(h2).any():
            raise ValueError("Inf values detected in h1 or h2 before normalization.")
        if num_negs is None:
            between_sim = self.f(self.cosine_similarity(h1, h2), tau)
            return -torch.log(between_sim.diag() / between_sim.sum(1))
        else:
            pos_sim = self.f(F.cosine_similarity(h1, h2), tau)
            negs = []
            for _ in range(num_negs):
                negs.append(h2[torch.randperm(h2.size(0))])
            negs = torch.stack(negs, dim=-1)
            neg_sim = self.f(F.cosine_similarity(h1.unsqueeze(-1).tile(num_negs), negs), tau)
            return -torch.log(pos_sim / (pos_sim + neg_sim.sum(1)))

    def __semi_loss_batch(self, h1: Tensor, h2: Tensor, tau: float, batch_size: int):
        if torch.isnan(h1).any() or torch.isnan(h2).any():
            raise ValueError("NaN values detected in h1 or h2 before normalization.")
        if torch.isinf(h1).any() or torch.isinf(h2).any():
            raise ValueError("Inf values detected in h1 or h2 before normalization.")
        device = h1.device
        num_samples = h1.size(0)
        num_batches = (num_samples - 1) // batch_size + 1
        indices = torch.arange(0, num_samples, device=device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]
            if torch.any(torch.norm(h1[mask], dim=1) == 0) or torch.any(torch.norm(h2, dim=1) == 0):
                raise ValueError("Zero vectors detected in h1 or h2 before cosine similarity.")
            # print("h1[mask] min:", h1[mask].min().item(), "max:", h1[mask].max().item())
            # print("h2 min:", h2.min().item(), "max:", h2.max().item())
            between_sim = self.f(self.cosine_similarity(h1[mask], h2), tau)

            loss = -torch.log(between_sim[:, i * batch_size: (i + 1) * batch_size].diag() / between_sim.sum(1))
            losses.append(loss)
        return torch.cat(losses)

    def f(self, x, tau):
        if tau == 0:
            raise ValueError("Tau cannot be zero in the exponential function.")
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input 'x' contains NaN or infinite values.")
        return torch.exp(x / tau)
    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        eps = 1e-8
        z1_norm = torch.norm(z1, dim=1, p=2).clamp(min=eps)
        z2_norm = torch.norm(z2, dim=1, p=2).clamp(min=eps)
        z1 = z1 / z1_norm.unsqueeze(1)
        z2 = z2 / z2_norm.unsqueeze(1)
        if z1.shape[1] != z2.shape[1]:
            raise ValueError("Mismatched shapes between z1 and z2 in cosine_similarity.")
        similarity = torch.mm(z1, z2.t())
        similarity = torch.clamp(similarity, min=-1.0, max=1.0)
        if torch.isnan(similarity).any() or torch.isinf(similarity).any():
            raise ValueError("NaN or Inf values detected in similarity matrix.")
        return similarity

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    def disc_similarity(self, z1: Tensor, z2: Tensor):
        return torch.sigmoid(self.disc(z1, z2)).squeeze()

    def cal_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        self.tau = 0.5
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        eps = 1e-8
        denominator = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() + eps
        safe_ratio = torch.clamp(between_sim.diag() / denominator, min=1e-8)

        return -torch.log(safe_ratio)

    def loss_hyperedge_ada_maxmargin(self, Z1, Z2):
        h1 = Z1.T
        h2 = Z2.T
        l1 = self.cal_loss(h1, h2)
        l2 = self.cal_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret

    def node_level_loss(self, n1: Tensor, n2: Tensor, node_tau: float,
                        batch_size: Optional[int] = None, num_negs: Optional[int] = None,
                        mean: bool = True):
        # print(f"n1_filtered shape: {n1.shape}")
        # print(f"n2_filtered shape: {n2.shape}")
        loss = self.__loss(n1, n2, node_tau, batch_size, num_negs, mean)
        return loss

    def group_level_loss(self, e1: Tensor, e2: Tensor, edge_tau: float,
                         batch_size: Optional[int] = None, num_negs: Optional[int] = None,
                         mean: bool = True):
        loss = self.__loss(e1, e2, edge_tau, batch_size, num_negs, mean)
        return loss

    def membership_level_loss_with_clusters(self, n: Tensor, e: Tensor, hyperedge_index: Tensor, tau: float,
                                            cluster_assignments_n: Tensor, cluster_assignments_e: Tensor,
                                            batch_size: Optional[int] = None, mean: bool = True):
        """
        Modified membership level loss function to use precomputed cluster assignments and a subset of positive and negative samples.

        Args:
            n (Tensor): Node embeddings.
            e (Tensor): Edge embeddings.
            hyperedge_index (Tensor): Hyperedge index tensor.
            tau (float): Temperature parameter.
            cluster_assignments_n (Tensor): Precomputed cluster assignments for node embeddings.
            cluster_assignments_e (Tensor): Precomputed cluster assignments for edge embeddings.
            batch_size (Optional[int]): Batch size for processing large datasets.
            mean (bool): If True, average the loss; otherwise, sum it.
            num_pos_samples (Optional[int]): Number of positive samples to use. If None, use all positives.
            num_neg_samples (Optional[int]): Number of negative samples to use. If None, use all negatives.

        Returns:
            Tensor: Computed loss.
        """
        # Permute the edge and node embeddings for negative samples
        e_perm = e[torch.randperm(e.size(0))]
        n_perm = n[torch.randperm(n.size(0))]

        # Determine the number of available samples
        num_available_pos = hyperedge_index.shape[1]
        num_available_neg = min(e_perm.size(0), n_perm.size(0))

        if batch_size is None:
            # Without batching
            pos = self.f(self.disc_similarity(n[hyperedge_index[0]], e[hyperedge_index[1]]), tau)
            neg_n = self.f(self.disc_similarity(n[hyperedge_index[0]], e_perm[hyperedge_index[1] % e_perm.size(0)]),
                           tau)
            neg_e = self.f(self.disc_similarity(n_perm[hyperedge_index[0] % n_perm.size(0)], e[hyperedge_index[1]]),
                           tau)

            # Use cluster-based weights as pseudo-labels
            cluster_weights = (
                    cluster_assignments_n[hyperedge_index[0]] == cluster_assignments_e[hyperedge_index[1]]).float()
            loss_n = -torch.log(pos / (pos + neg_n)) * cluster_weights
            loss_e = -torch.log(pos / (pos + neg_e)) * cluster_weights
        else:
            # With batching
            num_samples = hyperedge_index.shape[1]
            num_batches = (num_samples - 1) // batch_size + 1
            indices = torch.arange(0, num_samples, device=n.device)
            aggr_pos = []
            aggr_neg_n = []
            aggr_neg_e = []
            aggr_weights = []
            for i in range(num_batches):
                mask = indices[i * batch_size: (i + 1) * batch_size]
                if len(mask) > 0:
                    pos = self.f(self.disc_similarity(n[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]),
                                 tau)
                    if pos.shape != ():
                        neg_n = self.f(
                            self.disc_similarity(n[hyperedge_index[:, mask][0]],
                                                 e_perm[hyperedge_index[:, mask][1] % e_perm.size(0)]),
                            tau)
                        neg_e = self.f(
                            self.disc_similarity(n_perm[hyperedge_index[:, mask][0] % n_perm.size(0)],
                                                 e[hyperedge_index[:, mask][1]]),
                            tau)

                        # Append results only if mask > 0 and pos is not 0-dimensional
                        aggr_pos.append(pos)
                        aggr_neg_n.append(neg_n)
                        aggr_neg_e.append(neg_e)

                        # Collect cluster weights
                        aggr_weights.append((cluster_assignments_n[hyperedge_index[:, mask][0]] ==
                                             cluster_assignments_e[hyperedge_index[:, mask][1]]).float())

            aggr_pos = torch.concat(aggr_pos)
            aggr_neg_n = torch.concat(aggr_neg_n)
            aggr_neg_e = torch.concat(aggr_neg_e)
            aggr_weights = torch.concat(aggr_weights)

            # Compute loss with cluster weighting
            loss_n = -torch.log(aggr_pos / (aggr_pos + aggr_neg_n))
            loss_e = -torch.log(aggr_pos / (aggr_pos + aggr_neg_e))

            # Apply cluster-based weighting (pseudo-labels)
            # Compute loss with cluster-based weights
            loss_n = -torch.log(aggr_pos / (aggr_pos + aggr_neg_n)) * aggr_weights
            loss_e = -torch.log(aggr_pos / (aggr_pos + aggr_neg_e)) * aggr_weights

        loss_n = loss_n[~torch.isnan(loss_n)]
        loss_e = loss_e[~torch.isnan(loss_e)]
        loss = loss_n + loss_e
        loss = loss.mean() if mean else loss.sum()

        return loss

    def membership_level_loss(self, n: Tensor, e: Tensor, hyperedge_index: Tensor, tau: float,
                              batch_size: Optional[int] = None, mean: bool = True,
                              num_samples: Optional[int] = None):
        """
        Calculate the membership-level loss.

        Parameters:
        - n: Tensor of node features.
        - e: Tensor of edge (hyperedge) features.
        - hyperedge_index: Tensor of indices representing hyperedges.
        - tau: Temperature parameter for the similarity function.
        - batch_size: Optional batch size for batch processing.
        - mean: Boolean indicating whether to average the loss.
        - num_samples: Number of samples to limit the processing to.

        Returns:
        - The membership-level loss.
        """

        # If num_samples is provided, limit the number of samples in hyperedge_index
        if num_samples is not None:
            num_samples = min(hyperedge_index.shape[1], num_samples)
            perm = torch.randperm(hyperedge_index.shape[1])[:num_samples]
            hyperedge_index = hyperedge_index[:, perm]

        # Permute edges and nodes for creating negative samples
        e_perm = e[torch.randperm(e.size(0))]
        n_perm = n[torch.randperm(n.size(0))]

        if batch_size is None:
            # Process without batching
            pos = self.f(self.disc_similarity(n[hyperedge_index[0]], e[hyperedge_index[1]]), tau)
            neg_n = self.f(self.disc_similarity(n[hyperedge_index[0]], e_perm[hyperedge_index[1]]), tau)
            neg_e = self.f(self.disc_similarity(n_perm[hyperedge_index[0]], e[hyperedge_index[1]]), tau)

            # Calculate losses for nodes and edges
            loss_n = -torch.log(pos / (pos + neg_n))
            loss_e = -torch.log(pos / (pos + neg_e))
        else:
            # Process with batching
            num_samples = hyperedge_index.shape[1]
            num_batches = (num_samples - 1) // batch_size + 1
            indices = torch.arange(0, num_samples, device=n.device)

            aggr_pos = []
            aggr_neg_n = []
            aggr_neg_e = []
            for i in range(num_batches):
                mask = indices[i * batch_size: (i + 1) * batch_size]
                if len(mask) > 0:
                    pos = self.f(self.disc_similarity(n[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]),
                                 tau)
                    if pos.shape != ():
                        neg_n = self.f(
                            self.disc_similarity(n[hyperedge_index[:, mask][0]], e_perm[hyperedge_index[:, mask][1]]),
                            tau)
                        neg_e = self.f(
                            self.disc_similarity(n_perm[hyperedge_index[:, mask][0]], e[hyperedge_index[:, mask][1]]),
                            tau)

                        # Append results only if mask > 0 and pos is not 0-dimensional
                        aggr_pos.append(pos)
                        aggr_neg_n.append(neg_n)
                        aggr_neg_e.append(neg_e)

            # Concatenate aggregated results
            aggr_pos = torch.cat(aggr_pos)
            aggr_neg_n = torch.cat(aggr_neg_n)
            aggr_neg_e = torch.cat(aggr_neg_e)

            # Calculate losses for nodes and edges
            loss_n = -torch.log(aggr_pos / (aggr_pos + aggr_neg_n))
            loss_e = -torch.log(aggr_pos / (aggr_pos + aggr_neg_e))

        # Remove NaN values from the losses
        loss_n = loss_n[~torch.isnan(loss_n)]
        loss_e = loss_e[~torch.isnan(loss_e)]

        # Aggregate the loss values
        loss = loss_n + loss_e
        return loss.mean() if mean else loss.sum()

    def chgnn(self, data, args, view_gen1, view_gen2, model):
        """
        genSim + hypergraph contrast pipeline.

        Steps:
          1) Generate two augmented views via view generators.
          2) Encourage view-consistency with an MSE-based similarity term.
          3) Run encoder on each view and compute a symmetric hyperedge
             contrastive loss (loss_hyperedge_ada_maxmargin).
          4) Return total loss + view samples and indices.

        Args:
            data: PyG data object with x, edge_index, etc.
            args: Namespace with `device` and weighting hyperparams (e.g., w_sim).
            view_gen1: First HypergraphViewGenerator.
            view_gen2: Second HypergraphViewGenerator.
            model: Encoder model to produce embeddings for each view.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
                (gen_loss, sample1, edge_index1, sample2, edge_index2)
        """
        sample1, edge_index1 = view_gen1(data, args)
        sample2, edge_index2 = view_gen2(data, args)
        loss_sim = F.mse_loss(sample1, sample2)
        args.w_sim = 1.0
        loss_sim = args.w_sim * (1 - loss_sim).clone()
        data1 = data.clone().to(args.device)
        data2 = data.clone().to(args.device)
        data1.edge_index = edge_index1
        data2.edge_index = edge_index2
        edge_size=data.totedges
        node_size = data.n_x
        Z1, _, _, _ = model.forward(data1, edge_weight=None, edge_size=edge_size, node_size=node_size)
        Z2, _, _, _ = model.forward(data2, edge_weight=None,edge_size=edge_size, node_size=node_size)

        # hypergraph cluster contrast loss
        gen_loss = model.loss_hyperedge_ada_maxmargin(Z1, Z2) + loss_sim
        return gen_loss, sample1, edge_index1, sample2, edge_index2

    def pretrain_forward(self, n1, n2, e1, e2, edge_mask, edge_mask1, edge_mask2, masked_index1, masked_index2,
                         args, view_gen1, view_gen2, data, num_nodes, num_edges, num_negs, model, supervised=0):
        """
        Compute and aggregate pretraining losses for selected tasks.

        This wrapper orchestrates multiple self-supervised (and optional supervised)
        objectives during pretraining. Given two augmented views (n1/e1 and n2/e2),
        it conditionally computes:
          • 'genSim'       – view-consistency + hyperedge contrast via `chgnn`
          • 'node'         – node-level contrastive loss between (n1, n2)
          • 'graph'        – hyperedge-level contrastive loss between (e1, e2)
          • 'membership'   – membership contrast using masked incidence indices
          • 'supervised'   – pass-through supervised loss if supplied

        Args:
            n1, n2 (Tensor): Node embeddings from the two augmented views (|V|, D).
            e1, e2 (Tensor): Hyperedge embeddings from the two augmented views (|E|, D).
            edge_mask (BoolTensor): Valid hyperedge mask shared by both views (|E|,).
            edge_mask1 (BoolTensor): Valid hyperedge mask for view 1 (|E|,).
            edge_mask2 (BoolTensor): Valid hyperedge mask for view 2 (|E|,).
            masked_index1 (LongTensor): Masked incidence for view 1 (2, M1) with [node_idx; edge_idx].
            masked_index2 (LongTensor): Masked incidence for view 2 (2, M2) with [node_idx; edge_idx].
            args (Namespace): Hyperparameters, including temperatures and batch sizes.
            view_gen1, view_gen2: View generators used inside `chgnn` when 'genSim' is active.
            data: PyG data object used for encoding within `chgnn`.
            num_nodes (int): Number of nodes (used by downstream calls).
            num_edges (int): Number of hyperedges (used by downstream calls).
            num_negs (Optional[int]): If provided, number of sampled negatives in contrastive loss.
            model: Encoder model (used by `chgnn`).
            supervised (Tensor|float, optional): Supervised loss term (already computed) to include.

        Returns:
            Dict[str, Tensor]: A mapping from task name to its loss tensor, e.g.,
                {
                  'genSim': ...,
                  'node': ...,
                  'graph': ...,
                  'membership': ...,
                  'supervised': ...
                }
        """

        res = {}

        # 1) genSim: generate two views, enforce view similarity + hyperedge contrast
        if 'genSim' in args.tasks:
            res['genSim'], _, _, _, _ = self.chgnn(data=data, args=args, view_gen1=view_gen1, view_gen2=view_gen2,
                                                   model=model, num_nodes=num_nodes, num_edges=num_edges)
        # 2) Node-level contrast
        if 'node' in args.tasks:
            res['node'] = self.node_level_loss(n1, n2, args.pretrain_tau_n, batch_size=args.pretrain_ng_batch_size,
                                               num_negs=num_negs)
        # 3) Hyperedge-level (graph-level) contrast
        if 'graph' in args.tasks:
            res['graph'] = self.group_level_loss(e1[edge_mask], e2[edge_mask], args.pretrain_tau_g,
                                                 batch_size=args.pretrain_ng_batch_size,
                                                 num_negs=num_negs)
        # 4) Membership-level contrast: positive pairs come from masked incidences
        if 'membership' in args.tasks:
            loss_m1 = self.membership_level_loss(
                n=n1,
                e=e2[edge_mask2],
                hyperedge_index=masked_index2,
                tau=args.pretrain_tau_m,
                batch_size=args.pretrain_m_batch_size,
                mean=True
            )

            loss_m2 = self.membership_level_loss(
                n=n2,
                e=e1[edge_mask1],
                hyperedge_index=masked_index1,
                tau=args.pretrain_tau_m,
                batch_size=args.pretrain_m_batch_size,
                mean=True
            )
            res['membership'] = (loss_m1 + loss_m2) * 0.5
        # 5) Optional supervised loss passthrough
        if 'supervised' in args.tasks:
            res['supervised'] = supervised
        return res;

    def concatenate_tensors(self, tensor_list):
        """
        Concatenate tensors from a list while handling zero-dimensional tensors.
        """
        # Initialize an empty list for concatenation
        tensors_to_concatenate = []

        # Iterate through the tensor list
        for tensor in tensor_list:
            # If the tensor is not zero-dimensional, add it to the list for concatenation
            if tensor.dim() > 0:
                tensors_to_concatenate.append(tensor)
            else:
                # Skip zero-dimensional tensors or replace them with an empty tensor of appropriate shape
                # Alternatively, you can choose to log a warning or handle the zero-dimensional tensor in another way
                pass

        # If there are tensors to concatenate, perform the concatenation
        if tensors_to_concatenate:
            concatenated_tensor = torch.concat(tensors_to_concatenate)
            return concatenated_tensor

        # If no tensors to concatenate, return an empty tensor of appropriate type and device
        else:
            # Return an empty tensor with the same dtype and device as the first tensor in the original list
            if tensor_list:
                dtype = tensor_list[0].dtype
                device = tensor_list[0].device
                return torch.tensor([], dtype=dtype, device=device)
            else:
                return torch.tensor([])


class ViewLearner(torch.nn.Module):
    """
    Learns edge dropout/retention logits conditioned on (node, hyperedge) embeddings
    produced by the encoder. Outputs edge-wise logits for view construction.
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
