import torch
import torch, numpy as np, scipy.sparse as sp
import torch_sparse
from torch_scatter import scatter

def aug_node(overlappness,args,device):


    cut_off = 0.9
    p = 0.2

    weights = overlappness.clone()


    weights = (weights.max() -weights) / (weights.max()-weights.mean())
    if p<0. or p>1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))

    weights = weights * p
    weights = weights.where(weights<cut_off, torch.ones_like(weights)*cut_off)
    sel_mask = ~torch.bernoulli(1. - weights).to(torch.bool).to(device)

    return sel_mask


#edge_perturbation

def aug_edge(H,overlappness,args):

    cut_off = args.cut_off_edge
    p = args.edge_perturbation_rate

    (V, E), value = torch_sparse.from_scipy(H)
    edge_weight = scatter(overlappness[V],E, dim=0, reduce='mean')
    edge_weight = torch.log(edge_weight)


    edge_weight = (edge_weight.max()-edge_weight)/(edge_weight.max()-edge_weight.mean())

    if p<0. or p>1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))

    weights = edge_weight * p
    weights = weights.where(weights<cut_off, torch.ones_like(weights)*cut_off)
    sel_mask = ~torch.bernoulli(1. - weights).to(torch.bool)
    H = H.toarray().T
    H[sel_mask]  = np.zeros(H.shape[1])
    H = H.T
    H = sp.csr_matrix(H)
    return H
