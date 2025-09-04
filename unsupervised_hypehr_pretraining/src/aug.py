"""
============================================================
Augmentation Utilities for Hypergraph Learning
============================================================
This file implements node-level and edge-level augmentation
strategies used in hypergraph contrastive learning. It includes:

1. Node-level augmentation (`aug_node`):
   - Randomly drops nodes with probability inversely related
     to their 'overlappness' score.

2. Edge-level augmentation (`aug_edge`):
   - Randomly perturbs hyperedges in the incidence matrix
     based on node overlappness statistics.

Both functions aim to generate diverse views of the hypergraph
for robust self-supervised training.

----------------------------
Usage
Import and call directly in training or pretraining scripts:
    from aug import aug_node, aug_edge
"""



import torch, numpy as np, scipy.sparse as sp
import torch_sparse
from torch_scatter import scatter

def aug_node(overlappness,args,device):
    """
        Node-level augmentation.

        Randomly drops nodes based on their 'overlappness' score.
        Nodes with higher overlappness are kept with higher probability,
        while less informative nodes are dropped more often.

        Args:
            overlappness (Tensor): Node-wise scores (higher = more overlap).
            args (Namespace): Argument container (not used here, but included for consistency).
            device (torch.device): Device for returned mask.

        Returns:
            Tensor (bool): Selection mask indicating which nodes are kept.
        """

    cut_off = 0.9  # Maximum dropout probability
    p = 0.2  # Base dropout rate

    weights = overlappness.clone()
    # Normalize weights so that higher overlappness → lower drop prob
    weights = (weights.max() - weights) / (weights.max() - weights.mean())

    if p<0. or p>1.:
        raise ValueError(f"Dropout probability must be between 0 and 1, got {p}")

    # Scale by base dropout rate
    weights = weights * p
    # Clip probabilities at cut_off
    weights = weights.where(weights < cut_off, torch.ones_like(weights) * cut_off)

    # Sample dropout mask (True = drop)
    sel_mask = ~torch.bernoulli(1. - weights).to(torch.bool).to(device)
    return sel_mask


def aug_edge(H,overlappness,args):
    """
        Edge-level augmentation.

        Randomly perturbs hyperedges in incidence matrix H, based on
        edge weights derived from node overlappness.

        Args:
            H (scipy.sparse matrix): Hypergraph incidence matrix (nodes × hyperedges).
            overlappness (Tensor): Node-wise overlappness scores.
            args (Namespace): Must define:
                - cut_off_edge (float): Max dropout probability for edges.
                - edge_perturbation_rate (float): Base dropout rate.

        Returns:
            scipy.sparse.csr_matrix: Perturbed incidence matrix.
        """
    cut_off = args.cut_off_edge
    p = args.edge_perturbation_rate

    # Convert to COO indices (V = node indices, E = edge indices)
    (V, E), value = torch_sparse.from_scipy(H)

    # Compute edge weights as mean overlappness of its nodes
    edge_weight = scatter(overlappness[V],E, dim=0, reduce='mean')
    edge_weight = torch.log(edge_weight)

    # Normalize so edges with lower mean overlappness get higher drop prob
    edge_weight = (edge_weight.max()-edge_weight)/(edge_weight.max()-edge_weight.mean())

    if p<0. or p>1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))

    # Scale and clip probabilities
    weights = edge_weight * p
    weights = weights.where(weights<cut_off, torch.ones_like(weights)*cut_off)

    # Sample dropout mask (True = drop edge)
    sel_mask = ~torch.bernoulli(1. - weights).to(torch.bool)

    # Zero out dropped edges in incidence matrix
    H = H.toarray().T
    H[sel_mask]  = np.zeros(H.shape[1])
    H = H.T
    H = sp.csr_matrix(H)
    return H
