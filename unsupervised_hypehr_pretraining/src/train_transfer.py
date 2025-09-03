import os

import argparse
import csv
from tqdm import tqdm, trange
import copy
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score
import torch
from torch import Tensor
from torch_geometric.typing import Adj, Size, OptTensor
import random
from torch_scatter import scatter_add, scatter
from itertools import permutations

from generate_hypergraph import HypergraphViewGenerator
from models import *
from preprocessing import *
from convert_datasets_to_pygDataset import dataset_Hypergraph
import min_norm_solvers
from weight_methods import (
    METHODS,
    MGDA,
    STL,
    LinearScalarization,
    # NashMTL,
    PCGrad,
    Uncertainty, WeightMethods,
)


def generate_multiple_splits(label, num_splits=5, train_prop=.5, valid_prop=.25, ignore_negative=False, balance=False,
                             rand_seed=0):
    """
    Create multiple (train/valid/test) splits for cross-validation.

    Args:
        label (Tensor): Ground-truth labels.
        num_splits (int): Number of splits to generate.
        train_prop (float): Train proportion (0–1).
        valid_prop (float): Valid proportion (0–1).
        ignore_negative (bool): If True, ignore negatives when sampling.
        balance (bool): If True, balance classes in splits.
        rand_seed (int): Base seed; each split uses (base + split_id).

    Returns:
        dict[int, dict[str, Tensor]]: split_id → {'train','valid','test'} indices.
    """

    data_splits = {}

    for split in range(num_splits):
        split_idx = rand_train_test_idx(label, train_prop, valid_prop, ignore_negative, balance, rand_seed=split)
        data_splits[split] = split_idx  # Store the train/valid/test indices for each split

    return data_splits


def pretrain(num_negs, tasks, weighted_method, view_gen1, view_gen2):
    """
    Pretraining routine across self-supervised tasks.

    Supports tasks in `tasks`: {'genSim','node','graph','membership'} and
    (optionally) Pareto MTL (min-norm) or weighted methods.

    Args:
        num_negs: Negative samples for contrastive terms (or None).
        tasks (List[str]): Which pretrain tasks to optimize.
        weighted_method: WeightMethods instance when args.weighted_methods=True.
        view_gen1, view_gen2: Optional view generators for 'genSim'.

    Returns:
        Tensor: Scalar loss used for optimizer.step().

    Notes:
        - Uses globals: model, data, args, (and optionally criterion).
        - May call backward() inside (Pareto/weighted branches).
        - Applies grad clipping where appropriate.
    """

    loss_data = {}
    grads = {}
    model.zero_grad()

    features, hyperedge_index = data.x, data.edge_index
    cidx = hyperedge_index[1].min()
    hyperedge_index[1] -= cidx
    model.train()
    model.zero_grad()
    n1 = None
    n2 = None
    e1 = None
    e2 = None
    edge_mask = None
    edge_mask1 = None
    edge_mask2 = None
    masked_index1 = None
    masked_index2 = None
    if any(task in tasks for task in ['node', 'membership', 'graph']):
        hyperedge_index1 = drop_incidence(hyperedge_index, args.pretrain_drop_incidence_rate)
        hyperedge_index2 = drop_incidence(hyperedge_index, args.pretrain_drop_incidence_rate)
        x1 = drop_features(features, args.pretrain_drop_feature_rate)
        x2 = drop_features(features, args.pretrain_drop_feature_rate)

        node_mask1, edge_mask1 = valid_node_edge_mask(hyperedge_index1, args.num_nodes, data.totedges)
        node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, args.num_nodes, data.totedges)
        # node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, args.num_nodes, args.num_edges)
        node_mask = node_mask1 & node_mask2
        edge_mask = edge_mask1 & edge_mask2
        # edgeMaskShape = edge_mask.shape

        device = data.x.device

        data1 = data.clone().to(device)
        data1.x = x1
        data1.edge_index = hyperedge_index1

        data2 = data.clone()
        data2.x = x2
        data2.edge_index = hyperedge_index2
        edge_size = data.totedges
        _, e1, n1, _ = model.forward(data1, args, edge_mask1, edge_size=edge_size, node_size=data.n_x)
        _, e2, n2, _ = model.forward(data2, args, edge_mask2, edge_size=edge_size, node_size=data.n_x)
        e1, e2 = torch.sigmoid(e1), torch.sigmoid(e2)
        n1, n2 = torch.sigmoid(n1), torch.sigmoid(n2)

    if args.pareto:
        masked_index1 = None
        masked_index2 = None
        if 'genSim' in args.tasks:
            model.train()
            gen_loss, _, _, _, _ = model.chgnn(data, args, view_gen1, view_gen2, model)
            grads['genSim'] = []
            loss_data['genSim'] = gen_loss
            gen_loss.backward(retain_graph=True)
            for param in model.parameters():
                if param.grad is not None:
                    # print(f"Gradient for {param} before detaching in genSim: {param.grad}")
                    grads['genSim'].append(param.grad.data.detach().cpu())
                else:
                    grads['genSim'].append(torch.zeros_like(param.data).cpu())
            model.zero_grad()
            print('gen_loss:', gen_loss.item())
        if 'node' in tasks:
            loss_n = model.node_level_loss(n1, n2, args.pretrain_tau_n, batch_size=args.pretrain_ng_batch_size,
                                           num_negs=num_negs)
            assert not torch.isnan(n1).any(), "NaN detected in n1"
            assert not torch.isnan(n2).any(), "NaN detected in n2"
            grads['node'] = []
            loss_data['node'] = loss_n
            loss_n.backward(retain_graph=True)
            for param in model.parameters():
                if param.grad is not None:
                    grads['node'].append(param.grad.data.detach().cpu())
                else:
                    grads['node'].append(torch.zeros_like(param.data).cpu())
            model.zero_grad()
        if 'graph' in tasks:
            assert not torch.isnan(e1[edge_mask]).any(), "NaN detected in e1"
            assert not torch.isnan(e2[edge_mask]).any(), "NaN detected in e2"
            loss_g = model.group_level_loss(e1[edge_mask], e2[edge_mask], args.pretrain_tau_g,
                                            batch_size=args.pretrain_ng_batch_size,
                                            num_negs=num_negs)
            grads['graph'] = []
            loss_data['graph'] = loss_g
            loss_g.backward(retain_graph=True)
            for param in model.parameters():
                if param.grad is not None:
                    # print(f"Gradient for {param} before detaching: {param.grad}")
                    grads['graph'].append(param.grad.data.detach().cpu())
                else:
                    grads['graph'].append(torch.zeros_like(param.data).cpu())
            model.zero_grad()
        if 'membership' in tasks:
            num_nodes = int(hyperedge_index[0].max()) + 1
            num_edges = int(hyperedge_index[1].max()) + 1
            assert not torch.isnan(hyperedge_index).any(), "NaN detected in hyperedge_index"
            assert not torch.isnan(hyperedge_index).any(), "NaN detected in hyperedge_index"
            masked_index1 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask,
                                                    edge_mask1)
            masked_index2 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask,
                                                    edge_mask2)
            loss_m1 = model.membership_level_loss(
                n=n1,
                e=e2,
                hyperedge_index=masked_index2,
                tau=args.pretrain_tau_m,
                batch_size=args.pretrain_m_batch_size,
                mean=True
            )

            loss_m2 = model.membership_level_loss(
                n=n2,
                e=e1,
                hyperedge_index=masked_index1,
                tau=args.pretrain_tau_m,
                batch_size=args.pretrain_m_batch_size,
                mean=True
            )
            loss_m = (loss_m1 + loss_m2) * 0.5
            grads['membership'] = []
            loss_data['membership'] = loss_m
            loss_m.backward(retain_graph=True)
            for param in model.parameters():
                if param.grad is not None:
                    # print(f"Gradient for {param} before detaching: {param.grad}")
                    grads['membership'].append(param.grad.data.detach().cpu())
                else:
                    grads['membership'].append(torch.zeros_like(param.data).cpu())
            model.zero_grad()
        if len(tasks) > 1:
            if 'genSim' in grads:
                for idx, tensor in enumerate(grads['genSim']):
                    if torch.isnan(tensor).any():
                        print(f"NaN found in genSim gradient at index {idx}")
            gn = min_norm_solvers.gradient_normalizers(grads, loss_data, args.grad_norm)
            for t in loss_data:
                for gr_i in range(len(grads[t])):
                    if torch.isnan(grads[t][gr_i]).any():
                        print(f"NaN detected in grads[{t}][{gr_i}]")
                    if torch.isnan(gn[t]).any():
                        print(f"NaN detected in gn[{t}]")
                    grads[t][gr_i] = grads[t][gr_i] / gn[t].to(grads[t][gr_i].device)
            sol, _ = min_norm_solvers.MinNormSolver.find_min_norm_element_FW([grads[t] for t in tasks])
            sol = {k: sol[i] for i, k in enumerate(tasks)}
        else:
            sol = {tasks[0]: 1.}
        # -------------- End of Pareto Multi-Tasking Learning --------------
        model.zero_grad()
        train_loss = torch.tensor(0.0, device=args.device, requires_grad=True)
        actual_loss = torch.tensor(0.0, device=args.device)
        loss_dict = model.pretrain_forward(n1, n2, e1, e2, edge_mask, edge_mask1, edge_mask2, masked_index1,
                                           masked_index2, args, view_gen1, view_gen2, data, args.num_nodes,
                                           args.num_edges, num_negs, model)
        for i, l in loss_dict.items():
            print(f"sol: {sol[i]}, current loss: {l.item() if l is not None else 'None'}")
            train_loss = train_loss + float(sol[i]) * l
            # print('cur_train_loss:',train_loss.item())
            actual_loss += l

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        loss_dict['train_loss'] = actual_loss.detach()
        return train_loss
    elif args.weighted_methods:
        losses = []

        if 'genSim' in args.tasks:
            gen_loss, _, _, _, _ = model.chgnn(data, args, view_gen1, view_gen2, model)
            losses.append(gen_loss)
            loss_data['genSim'] = gen_loss
        if 'node' in tasks:
            loss_n = model.node_level_loss(n1, n2, args.pretrain_tau_n, batch_size=args.pretrain_ng_batch_size,
                                           num_negs=num_negs)
            losses.append(loss_n)
            loss_data['node'] = loss_n

        if 'graph' in tasks:
            loss_g = model.group_level_loss(e1[edge_mask], e2[edge_mask], args.pretrain_tau_g,
                                            batch_size=args.pretrain_ng_batch_size, num_negs=num_negs)
            losses.append(loss_g)
            loss_data['graph'] = loss_g

        if 'membership' in tasks:
            num_nodes = int(hyperedge_index[0].max()) + 1
            num_edges = int(hyperedge_index[1].max()) + 1
            masked_index1 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask,
                                                    edge_mask1)
            masked_index2 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask,
                                                    edge_mask2)
            # Limit both positive and negative samples to 200
            loss_m1 = model.membership_level_loss(
                n=n1,
                e=e2,
                hyperedge_index=masked_index2,
                tau=args.pretrain_tau_m,
                batch_size=args.pretrain_m_batch_size,
                mean=True
            )

            loss_m2 = model.membership_level_loss(
                n=n2,
                e=e1,
                hyperedge_index=masked_index1,
                tau=args.pretrain_tau_m,
                batch_size=args.pretrain_m_batch_size,
                mean=True
            )
            loss_m = (loss_m1 + loss_m2) * 0.5
            losses.append(loss_m)
            loss_data['membership'] = loss_m
        if 'supervised' in tasks:
            out_score_logits, _, _, weight_tuple = model(data)
            out = torch.sigmoid(out_score_logits)
            loss_supervised = criterion(out[train_idx], data.y[train_idx]) + args.view_lambda * torch.mean(
                weight_tuple[1].reshape(-1))
            loss_supervised.backward()
            grads['supervised'] = []
            loss_data['supervised'] = loss_supervised
            for param in model.parameters():
                if param.grad is not None:
                    grads['supervised'].append(param.grad.data.detach().cpu())
                else:
                    grads['supervised'].append(torch.zeros_like(param.data).cpu())
            model.zero_grad()
        shared_parameters = get_used_parameters(losses, model)
        # Perform backward pass using the WeightMethods class
        loss, extra_outputs = weight_method.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            representation=features,
        )
        return loss;
    if 'genSim' in args.tasks:
        gen_loss, _, _, _, _ = model.chgnn(data, args, view_gen1, view_gen2, model)
    else:
        gen_loss = 0

    total_loss = gen_loss * args.pretrain_w_gS
    if 'node' in args.tasks:
        loss_n = model.node_level_loss(n1, n2, args.pretrain_tau_n, batch_size=args.pretrain_ng_batch_size,
                                       num_negs=num_negs)
        total_loss += loss_n
        loss_data['node'] = loss_n
    if 'graph' in args.tasks:
        loss_g = model.group_level_loss(e1[edge_mask], e2[edge_mask], args.pretrain_tau_g,
                                        batch_size=args.pretrain_ng_batch_size,
                                        num_negs=num_negs)
        total_loss += args.pretrain_w_g * loss_g
        loss_data['graph'] = loss_g

    if 'membership' in args.tasks:
        num_nodes = int(hyperedge_index[0].max()) + 1
        num_edges = int(hyperedge_index[1].max()) + 1
        masked_index1 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask,
                                                edge_mask1)
        masked_index2 = hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask,
                                                edge_mask2)
        loss_m1 = model.membership_level_loss(
            n=n1,
            e=e2,
            hyperedge_index=masked_index2,
            tau=args.pretrain_tau_m,
            batch_size=args.pretrain_m_batch_size,
            mean=True
        )

        loss_m2 = model.membership_level_loss(
            n=n2,
            e=e1,
            hyperedge_index=masked_index1,
            tau=args.pretrain_tau_m,
            batch_size=args.pretrain_m_batch_size,
            mean=True
        )
        loss_m = (loss_m1 + loss_m2) * 0.5
        total_loss += args.pretrain_w_m * loss_m
        loss_data['membership'] = loss_m
    loss_supervised = 0
    if args.semiSupervised:
        out_score_logits, _, _, weight_tuple = model(data)
        out = torch.sigmoid(out_score_logits)
        loss_supervised = criterion(out[train_idx], data.y[train_idx]) + args.view_lambda * torch.mean(
            weight_tuple[1].reshape(-1))

    print("pretrain_loss:", total_loss);
    total_loss.backward()
    return total_loss


def get_used_parameters(losses, model):
    used_parameters = set()  # Use a set to avoid duplicates
    for l in losses:
        grads = torch.autograd.grad(l, model.parameters(), retain_graph=True, allow_unused=True)
        for p, g in zip(model.parameters(), grads):
            if g is not None:
                used_parameters.add(p)
    return list(used_parameters)  # Convert back to a list


def drop_features(x: Tensor, p: float):
    """
       Column-wise feature dropout: zeroes each feature dim with prob p.

       Args:
           x (Tensor): Node features [N, F].
           p (float): Drop probability per feature dimension.

       Returns:
           Tensor: Features with a subset of columns zeroed.
       """
    device = x.device  # Get device from input tensor
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=device).uniform_(0, 1) < p
    x = x.clone().to(device)  # Ensure x is on the same device
    x[:, drop_mask] = 0
    return x


def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr: OptTensor, mask: Tensor):
    """
        Apply a boolean mask over incidence tuples.

        Args:
            row (Tensor): Node indices per incidence.
            col (Tensor): Edge indices per incidence.
            hyperedge_attr (OptTensor): Optional incidence attributes.
            mask (Tensor[bool]): Incidence mask.

        Returns:
            Tuple[Tensor, Tensor, OptTensor]: Filtered (row, col, attr).
        """
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]


def drop_incidence(hyperedge_index: Tensor, p: float = 0.2):
    """
        Randomly drop incidences (node–edge links) from the hypergraph.

        Args:
            hyperedge_index (Tensor[2, M]): Incidence list (row=node, col=edge).
            p (float): Drop probability per incidence (0–1).

        Returns:
            Tensor[2, M’]: Incidence list after random dropping.
        """
    device = hyperedge_index.device  # Get device from input tensor

    if p == 0.0:
        return hyperedge_index

    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=device) >= p

    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)
    return hyperedge_index


def drop_nodes(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    """
        Randomly drop nodes and remove their incidences.

        Args:
            hyperedge_index (Tensor[2, M]): Incidence list.
            num_nodes (int): Number of nodes.
            num_edges (int): Number of hyperedges.
            p (float): Drop probability per node (0–1).

        Returns:
            Tensor[2, M’]: Incidence list after node removal.
        """


    device = hyperedge_index.device  # Get device from input tensor

    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_nodes, device=device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
                                hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges), device=device).to_dense()
    H[drop_idx, :] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def drop_hyperedges(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    """
        Randomly drop hyperedges and remove their incidences.

        Args:
            hyperedge_index (Tensor[2, M]): Incidence list.
            num_nodes (int): Number of nodes.
            num_edges (int): Number of hyperedges.
            p (float): Drop probability per hyperedge (0–1).

        Returns:
            Tensor[2, M’]: Incidence list after hyperedge removal.
        """


    device = hyperedge_index.device  # Get device from input tensor

    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_edges, device=device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
                                hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges), device=device).to_dense()
    H[:, drop_idx] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index

def valid_node_edge_mask(hyperedge_index: Tensor, num_nodes: int, num_hyperedge: int):
    """
        Compute masks of nodes/edges that remain incident to at least one link.

        Args:
            hyperedge_index (Tensor[2, M]): Incidence list.
            num_nodes (int): Number of nodes.
            num_hyperedge (int): Number of hyperedges.

        Returns:
            Tuple[Tensor(bool), Tensor(bool)]: (node_mask, edge_mask) where True means degree > 0.
        """
    device = hyperedge_index.device  # Get device from input tensor
    ones = hyperedge_index.new_ones(hyperedge_index.shape[1]).to(device)

    Dn = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)
    De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_hyperedge)
    node_mask = Dn != 0
    edge_mask = De != 0
    return node_mask, edge_mask


def common_node_edge_mask(hyperedge_indexs: list[Tensor], num_nodes: int, num_edges: int):
    """
       Intersect validity across multiple incidence sets (all views must be valid).

       Args:
           hyperedge_indexs (List[Tensor]): Multiple incidence lists.
           num_nodes (int): Number of nodes.
           num_edges (int): Number of hyperedges.

       Returns:
           Tuple[Tensor(bool), Tensor(bool)]: Nodes/edges valid in all views.
       """
    device = hyperedge_indexs[0].device  # Get device from input tensor
    hyperedge_weight = hyperedge_indexs[0].new_ones(num_edges).to(device)
    node_mask = hyperedge_indexs[0].new_ones((num_nodes,)).to(torch.bool, device=device)
    edge_mask = hyperedge_indexs[0].new_ones((num_edges,)).to(torch.bool, device=device)

    for index in hyperedge_indexs:
        Dn = scatter_add(hyperedge_weight[index[1]], index[0], dim=0, dim_size=num_nodes)
        De = scatter_add(index.new_ones(index.shape[1]), index[1], dim=0, dim_size=num_edges)
        node_mask &= Dn != 0
        edge_mask &= De != 0
    return node_mask, edge_mask


def hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask, edge_mask):
    """
       Keep only incidences whose node and edge are allowed by masks.

       Args:
           hyperedge_index (Tensor[2, M]): Incidence list.
           num_nodes (int): Total nodes (unused, for API symmetry).
           num_edges (int): Total hyperedges (unused, for API symmetry).
           node_mask (Tensor(bool) or None): Nodes to keep; None = keep all.
           edge_mask (Tensor(bool) or None): Edges to keep; None = keep all.

       Returns:
           Tensor[2, M’]: Filtered incidence list.
       """
    device = hyperedge_index.device  # Get device from input tensor
    if node_mask is None and edge_mask is None:
        return hyperedge_index

    # Get the rows (nodes) and columns (edges) from the sparse hyperedge_index
    row, col = hyperedge_index

    if node_mask is not None:
        # Only keep rows where node_mask is True
        node_mask_idx = torch.where(node_mask)[0].to(device)  # Indices of nodes that are kept
        row_mask = torch.isin(row, node_mask_idx)  # Mask to keep valid rows (nodes)
    else:
        row_mask = torch.ones(row.size(0), dtype=torch.bool, device=device)  # Keep all rows

    if edge_mask is not None:
        # Only keep columns where edge_mask is True
        edge_mask_idx = torch.where(edge_mask)[0].to(device)  # Indices of edges that are kept
        col_mask = torch.isin(col, edge_mask_idx)  # Mask to keep valid cols (edges)
    else:
        col_mask = torch.ones(col.size(0), dtype=torch.bool, device=device)  # Keep all columns

    # Apply the masks to row and col to filter the hyperedges
    mask = row_mask & col_mask
    filtered_hyperedge_index = hyperedge_index[:, mask]

    return filtered_hyperedge_index


def clique_expansion(hyperedge_index: Tensor):
    """
      Convert each hyperedge into a directed clique over its incident nodes.

      Args:
          hyperedge_index (Tensor[2, M]): Incidence list.

      Returns:
          Tensor[2, E]: Edge list of the expanded (directed) graph.
      """
    edge_set = set(hyperedge_index[1].tolist())
    adjacency_matrix = []
    for edge in edge_set:
        mask = hyperedge_index[1] == edge
        nodes = hyperedge_index[:, mask][0].tolist()
        for e in permutations(nodes, 2):
            adjacency_matrix.append(e)

    adjacency_matrix = list(set(adjacency_matrix))
    adjacency_matrix = torch.LongTensor(adjacency_matrix).T.contiguous()
    return adjacency_matrix.to(hyperedge_index.device)


def parse_method(args, data):
    model = None
    if args.dname in ['mimic3', 'cradle', 'promote'] or args.dname.startswith('combine_') or args.dname.startswith('separate_'):
        if args.simpleModel:
            model = SimpleHypergraphModel(in_channels=args.feature_dim, hidden_channels=args.MLP_hidden,
                                          out_channels=args.MLP_hidden, num_labels=args.num_labels)
        else:
            model = SetGNN(args, data)
    return model


# random seed
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def GNN_evaluator(model, X, hyperedge_index, Y, test_idx, data, eval_func, epoch, method, args, mode='dev',
                  threshold=0.5):
    """
        Evaluate model on a subset of indices using a dataset-specific metric fn.

        Args:
            model: Trained model.
            X (Tensor): Node features (unused; kept for API).
            hyperedge_index (Tensor): Incidence list (unused; kept for API).
            Y (Tensor): Ground-truth labels.
            test_idx (Tensor): Indices to evaluate on.
            data: Full Data object for forward pass.
            eval_func (callable): Metric function (mimic3/cradle).
            epoch (int): Current epoch.
            method (str): Model name for logs.
            args: Config with thresholds, etc.
            mode (str): Tag for the eval split (e.g., 'dev_g').
            threshold (float): Binarization threshold for sigmoid outputs.

        Returns:
            Tuple[float, float, float, float]: (acc, auc, aupr, f1_macro).
        """
    with torch.no_grad():
        model.eval()
        n_node, n_edge = torch.max(hyperedge_index[0]) + 1, torch.max(hyperedge_index[1]) + 1

        # Get predicted class labels
        out_score_logits, _, _, weight_tuple = model(data)
        out = torch.sigmoid(out_score_logits)
        # out=model(data)
        pred_label = out[test_idx]

        # Ground truth labels
        true_labels = Y[test_idx].long()

        # Calculate accuracy
        # acc = torch.sum(pred_label == true_labels).float() / float(pred_label.shape[0])

        acc_g, auc_g, aupr_g, f1_macro_g = eval_func(
            y_true=true_labels, y_pred=pred_label,
            epoch=epoch, method=method, args=args, mode='dev_g', threshold=threshold)

        return acc_g, auc_g, aupr_g, f1_macro_g


@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, epoch, method, dname, args):
    """
        Evaluate on original graph (G), factual (G') and counterfactual (G−G') views.

        Uses the model (and optionally a learned view_learner) to compute metrics
        on validation and test sets for each view.

        Args:
            model: Trained model.
            data: PyG Data object.
            split_idx (dict): {'train','valid','test'} index tensors.
            eval_func (callable): Dataset-specific metric function.
            epoch (int): Current epoch.
            method (str): Model name tag.
            dname (str): Dataset name.
            args: Config flags including thresholds and temperature.

        Returns:
            Tuple[...]: Metrics for valid/test across G, G', and G−G' (16 floats).
        """

    valid_acc_gf = valid_auc_gf = valid_aupr_gf = valid_f1_macro_gf = \
        test_acc_gf = test_auc_gf = test_aupr_gf = test_f1_macro_gf = \
        valid_acc_gcf = valid_auc_gcf = valid_aupr_gcf = valid_f1_macro_gcf = \
        test_acc_gcf = test_auc_gcf = test_aupr_gcf = test_f1_macro_gcf = 0

    model.eval()

    # use original graph (G)
    out_score_g_logits, edge_feat, node_feat, weight_tuple = model(data)
    out_g = torch.sigmoid(out_score_g_logits)

    valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g = eval_func(
        data.y[split_idx['valid']], out_g[split_idx['valid']],
        epoch, method, dname, args, mode='dev_g', threshold=args.threshold)
    test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g = eval_func(data.y[split_idx['test']],
                                                                     out_g[split_idx['test']],
                                                                     epoch, method, dname, args,
                                                                     mode='test_g',
                                                                     threshold=args.threshold)

    if args.vanilla:
        edge_index = weight_tuple[0]
        edge_weight = weight_tuple[1].reshape(-1)
        # num_hyperedges = data.num_hyperedges[0]
        num_hyperedges = data.num_hyperedges
    else:
        # get the edge weight
        view_learner.eval()
        weight_logits = view_learner(data, device)

        # gumbel softmax
        # temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(weight_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        gate_inputs = (gate_inputs + weight_logits) / args.temperature
        aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

        # use factual graph (G')
        out_score_gf_logits, _, _, _ = model(data, edge_weight=aug_edge_weight)  # use augmented graph
        out_gf = torch.sigmoid(out_score_gf_logits)

        valid_acc_gf, valid_auc_gf, valid_aupr_gf, valid_f1_macro_gf = eval_func(
            data.y[split_idx['valid']],
            out_gf[split_idx['valid']],
            epoch, method, dname, args, mode='dev_gf', threshold=args.threshold)
        test_acc_gf, test_auc_gf, test_aupr_gf, test_f1_macro_gf = eval_func(
            data.y[split_idx['test']], out_gf[split_idx['test']],
            epoch, method, dname, args, mode='test_gf', threshold=args.threshold)

        # use counterfactual graph (G-G')
        out_score_gcf_logits, _, _, _ = model(data, edge_weight=1 - aug_edge_weight)  # use augmented graph
        out_gcf = torch.sigmoid(out_score_gcf_logits)

        valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf = eval_func(
            data.y[split_idx['valid']],
            out_gcf[split_idx['valid']],
            epoch, method, dname, args,
            mode='dev_gcf', threshold=args.threshold)
        test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf = eval_func(
            data.y[split_idx['test']], out_gcf[split_idx['test']],
            epoch, method, dname, args,
            mode='test_gcf', threshold=args.threshold)

        if epoch == args.epochs - 1:
            get_subset_ranking(aug_edge_weight, data.edge_index, data.num_hyperedges, args)

    return valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g, \
        test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g, \
        valid_acc_gf, valid_auc_gf, valid_aupr_gf, valid_f1_macro_gf, \
        test_acc_gf, test_auc_gf, test_aupr_gf, test_f1_macro_gf, \
        valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf, \
        test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf


def get_subset_ranking(edge_weight, edge_index, num_hyperedges, args):
    """
        Rank node incidences per hyperedge by learned edge weights and save lists.

        Produces two files under outputs/:
            - remained_output_*: top-k kept nodes per hyperedge
            - deleted_output_*: remaining nodes per hyperedge
        where k = ceil(len(hyperedge) * remain_percentage) with a minimum of 5 if possible.

        Args:
            edge_weight (Tensor): Learned importance per incidence.
            edge_index (Tensor[2, M]): Incidence list.
            num_hyperedges (int): Number of hyperedges.
            args: Config with remain_percentage, method, dname, vanilla flag.
        """

    edge_index_clone = edge_index.clone().detach().to('cpu').numpy()
    edge_weight_clone = edge_weight.reshape(1, -1).clone().detach().to('cpu').numpy()
    index_weight_concat = np.concatenate((edge_index_clone, edge_weight_clone), axis=0)

    index_weight_concat = index_weight_concat[:, index_weight_concat[2, :].argsort()[::-1]]

    edge_dict = {}
    for i in range(num_hyperedges):
        edge_dict[i] = []
    for i in tqdm(range(index_weight_concat.shape[1])):
        if index_weight_concat[1][i] < num_hyperedges:  # self loop
            edge_dict[index_weight_concat[1][i]].append(index_weight_concat[0][i])
    sorted_edge_dict = dict(sorted(edge_dict.items()))

    vanilla = ""
    if args.vanilla: vanilla = "_vanilla"
    with open(f"outputs/deleted_output_{args.method}{vanilla}_{args.dname}.txt", "w") as f_del, \
            open(f"outputs/remained_output_{args.method}{vanilla}_{args.dname}.txt", "w") as f_rem:
        for hyperedge in list(sorted_edge_dict.values()):
            rem_size = int(len(hyperedge) * args.remain_percentage)
            if rem_size < 5 and len(hyperedge) >= 5:
                rem_size = 5
            elif rem_size < 5 and len(hyperedge) < 5:
                rem_size = len(hyperedge)
            remain = [str(int(x)) for x in hyperedge[:rem_size]]
            f_rem.write(",".join(remain))
            f_rem.write('\n')
            delete = [str(int(x)) for x in hyperedge[rem_size:]]
            f_del.write(",".join(delete))
            f_del.write('\n')


def eval_mimic3(y_true, y_pred, epoch, method, args, mode='dev', threshold=0.5):
    """
       Compute multilabel metrics for MIMIC-III-style tasks and log per-phenotype.

       Args:
           y_true (Tensor): Ground-truth labels (N×L).
           y_pred (Tensor): Sigmoid outputs (N×L).
           epoch (int): Current epoch (logged to CSV).
           method (str): Model name tag for logs.
           args: Config (uses num_labels).
           mode (str): 'dev' or 'test' log tag.
           threshold (float): Binarization threshold.

       Returns:
           Tuple[float, float, float, float]: (accuracy, ROC-AUC, AUPR, F1-macro).
       """

    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    pred = np.array(y_pred > threshold).astype(int)
    correct = (pred == y_true)

    total_acc = []
    total_f1 = []
    for i in range(args.num_labels):
        correct = (pred[:, i] == y_true[:, i])
        accuracy = correct.sum() / correct.size
        total_acc.append(accuracy)
        f1_macro = f1_score(y_true[:, i], pred[:, i], average='macro')
        total_f1.append(f1_macro)

    correct = (pred == y_true)
    accuracy = correct.sum() / correct.size
    f1_macro = f1_score(y_true, pred, average='macro')

    total_auc = []
    for i in range(args.num_labels):
        roc_auc = roc_auc_score(y_true[:, i].reshape(-1), y_pred[:, i].reshape(-1))
        total_auc.append(roc_auc)

    roc_auc = roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))

    total_aupr = []
    for i in range(args.num_labels):
        aupr = average_precision_score(y_true[:, i].reshape(-1), y_pred[:, i].reshape(-1))
        total_aupr.append(aupr)
    aupr = average_precision_score(y_true.reshape(-1), y_pred.reshape(-1))

    with open(f'outputs/mimic3_{mode}_{method}.csv', 'a+', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["Epoch", "Phenotype", "acc", "auc", 'aupr', 'f1'])
        for i, (acc_, auc_, aupr_, f1_) in enumerate(zip(total_acc, total_auc, total_aupr, total_f1)):
            write_lst = [epoch, f"Phenetype {i}", acc_, auc_, aupr_, f1_]
            writer.writerow(write_lst)

    return accuracy, roc_auc, aupr, f1_macro


def eval_cradle(y_true, y_pred, epoch, method, args, mode='dev', threshold=0.5):
    """
        Compute binary metrics for CRADLE/PROMOTE-style tasks.

        Args:
            y_true (Tensor): Ground-truth labels.
            y_pred (Tensor): Sigmoid outputs.
            epoch (int): Current epoch (unused for CSV here).
            method (str): Model name tag.
            args: Config object (unused).
            mode (str): 'dev' or 'test' tag.
            threshold (float): Binarization threshold.

        Returns:
            Tuple[float, float, float, float]: (accuracy, ROC-AUC, AUPR, F1-macro).
        """

    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    pred = np.array(y_pred > threshold).astype(int)
    correct = (pred == y_true)
    accuracy = correct.sum() / correct.size
    f1_macro = f1_score(y_true.reshape(-1), pred.reshape(-1), average="macro")
    roc_auc = roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))
    aupr = average_precision_score(y_true.reshape(-1), y_pred.reshape(-1))

    return accuracy, roc_auc, aupr, f1_macro


def int_or_none(value):
    if value.lower() == 'none':
        return None
    else:
        return int(value)


def float_or_none(value):
    if value is None or value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Value '{value}' cannot be converted to float or None")


if __name__ == '__main__':
    os.chdir('path_to_src')  # working dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.7)
    parser.add_argument('--valid_prop', type=float, default=0.1)
    parser.add_argument('--Pdname', default='separate_icd3_promote')
    parser.add_argument('--dname', default='separate_icd3_mimic4')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-3, type=float)
    parser.add_argument('--view_lr', default=1e-2, type=float)
    parser.add_argument('--view_wd', default=1e-3, type=float)
    # How many layers of full NLConvs
    parser.add_argument('--All_num_layers', default=1, type=int)
    parser.add_argument('--MLP_num_layers', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--MLP_hidden', default=32,
                        type=int)  # Encoder hidden units
    parser.add_argument('--Classifier_num_layers', default=2,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=30,
                        type=int)  # Decoder hidden units
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    # ['all_one','deg_hxalf_sym']
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_labels', default=1, type=int)  # set the default for now, 25 for mimic, 1 for promote
    parser.add_argument('--Pnum_nodes', default=2620, type=int)
    parser.add_argument('--num_nodes', default=2620, type=int)  # 7423 for mimic and 12725 for cradle, 2653 for promote
    # 'all' means all samples have labels, otherwise it indicates the first [num_labeled_data] rows that have the labels
    parser.add_argument('--num_labeled_data', default='all', type=str)  # mimic3='12353', 'all' for promote
    parser.add_argument('--feature_dim', default=128, type=int)  # feature dim of learnable node feat
    parser.add_argument('--LearnFeat', action='store_true')
    # whether the he contain self node or not
    parser.add_argument('--PMA', action='store_true')
    #     Args for Attentions
    parser.add_argument('--heads', default=4, type=int)  # Placeholder
    parser.add_argument('--output_ ', default=1, type=int)  # Placeholder

    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--view_alpha', type=float, default=0.5)
    parser.add_argument('--view_lambda', type=float, default=5)
    parser.add_argument('--model_lambda', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=1)  # 0.5 | 5; temperature for gumbel softmax

    parser.add_argument('--vanilla', action='store_true', default=True)
    parser.add_argument('--remain_percentage', default=0.3, type=float)
    parser.add_argument('--rand_seed', default=0, type=int)
    parser.add_argument('--method', default='AllSetTransformer', type=str)
    parser.add_argument('--pretrain', default=True, type=bool)
    parser.add_argument('--pretrain_epoch', default=1, type=int)
    parser.add_argument('--pretrain_weight_decay', default=1.0e-05, type=float)
    parser.add_argument('--pretrain_lr', default=1e-3, type=float)
    parser.add_argument('--pretrain_drop_incidence_rate', default=0.3, type=float)
    parser.add_argument('--pretrain_drop_feature_rate', default=0.3, type=float)
    parser.add_argument('--pretrain_tau_n', default=0.5, type=float)
    parser.add_argument('--pretrain_tau_g', default=0.5, type=float)
    parser.add_argument('--pretrain_tau_m', default=1.0, type=float)
    parser.add_argument('--pretrain_w_gS', default=4, type=float)
    parser.add_argument('--pretrain_w_g', default=4, type=float)
    parser.add_argument('--pretrain_w_m', default=1, type=float)
    parser.add_argument('--pretrain_w_s', default=1, type=float)
    parser.add_argument('--pretrain_ng_batch_size', default=100, type=int_or_none)
    parser.add_argument('--pretrain_m_batch_size', default=2048, type=int)
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=['genSim', 'node', 'graph', 'membership'])  # 'genSim','node', 'graph', 'membership'
    parser.add_argument('--grad_norm', type=str, default='l2', choices=['l2', 'loss', 'loss+', 'none'])
    parser.add_argument('--num_clusters', default=16, type=int)
    parser.add_argument('--pareto', default=False, type=bool)
    parser.add_argument('--weighted_methods', default=False, type=bool)
    parser.add_argument('--methods', default='nashmtl', type=str)
    parser.add_argument('--swav', default=False, type=bool)
    parser.add_argument('--semiSupervised', default=False, type=bool)
    parser.add_argument('--formalTrain', default=False, type=bool)
    parser.add_argument('--train_percentage', type=float, default=1.0)
    parser.add_argument('--pretrain_SGD', default=False, type=bool)
    parser.add_argument('--pretrain_SGD_momentum', type=float, default=0.9)
    parser.add_argument('-num_folds', '--num_folds', type=int, default=5)
    parser.add_argument('-simpleModel', '--simpleModel', type=bool, default=False)
    parser.set_defaults(PMA=True)
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(LearnFeat=False)

    args = parser.parse_args()
    print("Pnum_nodes:", args.Pnum_nodes)
    print("num_nodes:",args.num_nodes)

    existing_dataset = ['mimic3', 'cradle', 'promote', 'combine_icd3']

    synthetic_list = ['mimic3', 'cradle', 'promote', 'combine_icd3']

    dname = args.dname
    p2raw = '../data/raw_data/'
    # put things to device
    if args.cuda != '-1':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    # print("device",device);
    Pdname = args.Pdname
    p2raw = '../data/raw_data/'
    dataset = dataset_Hypergraph(name=Pdname, root='../data/pyg_data/hypergraph_dataset/',
                                 p2raw=p2raw, num_nodes=args.Pnum_nodes)
    data = dataset.data
    args.num_nodes = data.n_x
    args.num_edges = data.num_hyperedges
    args.num_features = dataset.num_features
    if args.Pdname in ['mimic3', 'cradle', 'promote'] or args.dname.startswith('separate_'):
        # Shift the y label to start with 0
        data.y = data.y - data.y.min()
    if not hasattr(data, 'n_x'):
        data.n_x = torch.tensor([data.x.shape[0]])
    if not hasattr(data, 'num_hyperedges'):
        # note that we assume the he_id is consecutive.
        data.num_hyperedges = torch.tensor(
            [data.edge_index[0].max() - data.n_x[0] + 1])

    if args.method == 'AllSetTransformer':
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        data = norm_contruction(data, option=args.normtype)
    data_splits = generate_multiple_splits(data.y, num_splits=args.num_folds, train_prop=args.train_prop,
                                           valid_prop=args.valid_prop)
    model = parse_method(args, data)
    view_learner = ViewLearner(parse_method(args, data), args.MLP_hidden)
    # put things to device
    if args.cuda != '-1':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    args.device = device
    model, view_learner, data = model.to(device), view_learner.to(device), data.to(device)
    criterion = nn.BCELoss()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr,
                                       weight_decay=args.pretrain_weight_decay)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr, weight_decay=args.view_wd)

    if dname in ['mimic3']:
        eval_function = eval_mimic3
    elif dname in ['cradle', 'promote'] or args.dname.startswith('separate_'):
        eval_function = eval_cradle

    # Pretraining Phase
    print("Starting Pretraining Phase")
    # Load dataset only filtered by percent_data
    # Load dataset only filtered by percent_data

    weight_method = None
    if args.weighted_methods:
        weight_method = WeightMethods(
            args.methods, n_tasks=len(args.tasks), device=device
        )
    # Pretrain the model
    if args.pretrain:
        best_model_params = None
        best_view_model_params = None
        best_gen1_model_params = None
        best_gen2_model_params = None
        view_gen1 = None
        view_gen2 = None
        best_loss = float("inf")
        model.reset_parameters()
        if 'genSim' in args.tasks:
            view_gen1 = HypergraphViewGenerator(args.feature_dim, 3).to(args.device)
            view_gen2 = HypergraphViewGenerator(args.feature_dim, 3).to(args.device)
            gen_optimizer = torch.optim.AdamW([{'params': view_gen1.parameters()},
                                               {'params': view_gen2.parameters()}],
                                              lr=1e-2,
                                              weight_decay=1e-3)
        min_loss = float('inf')
        valid_score = 0
        best_pertrain_epoch = 0
        best_edge_feat = None
        for epoch in trange(args.pretrain_epoch, desc='Pretrain Epoch'):
            model.train()
            model.zero_grad()
            if 'genSim' in args.tasks:
                view_gen1.zero_grad()
                view_gen2.zero_grad()
            with torch.no_grad():
                _, edge_feat, _, _ = model.forward(data)
            pretrain_loss = pretrain(num_negs=None, tasks=args.tasks, weighted_method=weight_method,
                                     view_gen1=view_gen1, view_gen2=view_gen2)
            print(pretrain_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model_optimizer.step()
            if 'genSim' in args.tasks:
                torch.nn.utils.clip_grad_norm_(view_gen1.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(view_gen2.parameters(), 1)
                gen_optimizer.step()
            if (pretrain_loss < min_loss):
                min_loss = pretrain_loss
                best_model_params = copy.deepcopy(model.state_dict())
                best_view_model_params = copy.deepcopy(view_learner.state_dict())
                best_gen1_model_params = copy.deepcopy(view_gen1.state_dict())
                best_gen2_model_params = copy.deepcopy(view_gen2.state_dict())
                best_pertrain_epoch = epoch

        dname = args.dname
        p2raw = '../data/raw_data/'
        dataset = dataset_Hypergraph(name=dname, root='../data/pyg_data/hypergraph_dataset/',
                                     p2raw=p2raw, num_nodes=args.num_nodes)
        data = dataset.data
        args.num_nodes = data.n_x
        args.num_edges = data.num_hyperedges
        args.num_features = dataset.num_features
        if args.dname in ['mimic3', 'cradle', 'promote'] or args.dname.startswith('separate_'):
            # Shift the y label to start with 0
            data.y = data.y - data.y.min()
        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            # note that we assume the he_id is consecutive.
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max() - data.n_x[0] + 1])

        if args.method == 'AllSetTransformer':
            data = ExtractV2E(data)
            if args.add_self_loop:
                data = Add_Self_Loops(data)
            data = norm_contruction(data, option=args.normtype)

        data=data.to(device)
        if best_model_params is not None:
            model.load_state_dict(best_model_params)
            view_learner.load_state_dict(best_view_model_params)
            view_gen1.load_state_dict(best_gen1_model_params)
            view_gen2.load_state_dict(best_gen2_model_params)
        min_loss = float('inf')
        valid_score = 0
        best_pertrain_epoch = 0
        output_file = "edge_representation/best_edge_representation.txt"
        best_edge_feat = None
        patience = 10
        epochs_since_improvement = 0
        for epoch in trange(args.pretrain_epoch, desc='Pretrain Epoch'):
            model.train()
            model.zero_grad()
            if 'genSim' in args.tasks:
                view_gen1.zero_grad()
                view_gen2.zero_grad()
            with torch.no_grad():
                _, edge_feat, _, _ = model.forward(data)
            pretrain_loss = pretrain(num_negs=None, tasks=args.tasks, weighted_method=weight_method,
                                     view_gen1=view_gen1, view_gen2=view_gen2)
            print(pretrain_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model_optimizer.step()
            if 'genSim' in args.tasks:
                torch.nn.utils.clip_grad_norm_(view_gen1.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(view_gen2.parameters(), 1)
                gen_optimizer.step()
            if (pretrain_loss < min_loss):
                min_loss = pretrain_loss
                best_pertrain_epoch = epoch
                best_edge_feat = edge_feat[:data.n_label].cpu().numpy()
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best pretraining loss: {min_loss:.4f}")
                break

        # print(f"Best validation auc: {valid_score:.4f} at epoch {best_ep}")
        print(f"Best pretrain loss: {min_loss:.4f} at epoch {best_pertrain_epoch}")
        if best_edge_feat is not None:
            with open(output_file, 'w') as f:
                # Write the edge representation to the file
                f.write(
                    f"{best_edge_feat.shape[0]} {best_edge_feat.shape[1]}\n")  # Write dimensions (e.g., num_edges x embedding_dim)
                for edge_idx, embedding in enumerate(best_edge_feat):
                    embedding_str = ' '.join(map(str, embedding))
                    f.write(f"{edge_idx} {embedding_str}\n")
        print(f"Best edge representation saved to {output_file} with minimum loss {min_loss}")

    if args.formalTrain:
        model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                           weight_decay=args.wd)
        view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr, weight_decay=args.view_wd)

        # Formal Training Phase
        print("Starting Formal Training Phase")

        edge_id_dict = None
        # Reset the model to the pretrained state
        results = []

        for seed in tqdm(range(5)):
            if args.pretrain:
                model.load_state_dict(best_model_params)
            else:
                model.reset_parameters()
            seed_everything(seed)
            split_data = data_splits[seed]
            train_idx = split_data['train']
            valid_idx = split_data['valid']
            test_idx = split_data['test']
            if args.train_percentage < 1.0:
                # Shuffle and select a subset of the labeled data
                labeled_data = train_idx.tolist()  # Convert to list
                random.shuffle(labeled_data)  # Shuffle the data

                # Compute the number of samples to use based on the given percentage
                subset_size = int(len(labeled_data) * args.train_percentage)

                # Select the subset
                train_idx = torch.tensor(labeled_data[:subset_size], dtype=torch.long)

            validation_frequency = 10
            # Retain the pretrained model and continue training
            with torch.autograd.set_detect_anomaly(True):
                valid_score = 0
                params = None  # Initialize the variable to store model state
                best_ep = 0
                for epoch in trange(args.epochs, desc='Train Epoch'):
                    if args.vanilla:  # VANILLA - Use attention weight to get an important set for each encounter
                        model.train()
                        model.zero_grad()
                        out_score_logits, _, _, weight_tuple = model(data)
                        out = torch.sigmoid(out_score_logits)
                        model_loss = criterion(out[train_idx], data.y[train_idx]) + args.view_lambda * torch.mean(
                            weight_tuple[1].reshape(-1))

                        print(f"model loss in formal train: {model_loss:.4f}")
                        model_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        model_optimizer.step()

                        if (epoch + 1) % validation_frequency == 0:
                            valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g = GNN_evaluator(model=model,
                                                                                                     X=data.x,
                                                                                                     hyperedge_index=data.edge_index,
                                                                                                     Y=data.y,
                                                                                                     test_idx=valid_idx,
                                                                                                     data=data,
                                                                                                     method=args.method,
                                                                                                     eval_func=eval_function,
                                                                                                     epoch=epoch,
                                                                                                     args=args)
                            if valid_auc_g > valid_score:
                                params = copy.deepcopy(model.state_dict())  # Save best model
                                valid_score = valid_auc_g
                                best_ep = epoch
                    else:  # CACHE
                        if (epoch + 1) % 50 == 0:
                            args.view_lambda *= 0.5
                        """STEP ONE - TRAIN THE LEARNER"""
                        view_learner.train()
                        view_learner.zero_grad()
                        model.eval()

                        out_score_logits, out_edge_feat, _, _ = model(data)
                        out = torch.sigmoid(out_score_logits)

                        weight_logits = view_learner(data, device)

                        # gumbel softmax
                        # temperature = 1.0
                        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
                        eps = (bias - (1 - bias)) * torch.rand(weight_logits.size()) + (1 - bias)
                        gate_inputs = torch.log(eps) - torch.log(1 - eps)
                        gate_inputs = gate_inputs.to(device)
                        gate_inputs = (gate_inputs + weight_logits) / args.temperature
                        aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

                        # factual prediction
                        out_score_f_logits, out_edge_feat_f, _, _ = model(data, edge_weight=aug_edge_weight)
                        out_f = torch.sigmoid(out_score_f_logits)

                        # regularization - not to drop too many edges
                        edge_dropout_prob = 1 - aug_edge_weight
                        reg = torch.mean(edge_dropout_prob)

                        # counterfactual prediction
                        out_score_cf_logits, out_edge_feat_cf, _, _ = model(data, edge_weight=edge_dropout_prob)
                        out_cf = torch.sigmoid(out_score_cf_logits)

                        # factual loss
                        coef = out.detach().clone()
                        coef[out >= 0.5] = 1
                        coef[out < 0.5] = -1
                        loss_f = torch.mean(torch.clamp(torch.add(coef * (0 - out_score_f_logits), args.gamma), min=0))

                        # counterfactual loss
                        coef = out.detach().clone()
                        coef[out >= 0.5] = -1
                        coef[out < 0.5] = 1
                        loss_cf = torch.mean(
                            torch.clamp(torch.add(coef * (0 - out_score_cf_logits), args.gamma), min=0))

                        # factual and counterfactual view loss
                        loss = args.view_alpha * loss_f + (1 - args.view_alpha) * loss_cf

                        view_loss = loss + args.view_lambda * torch.mean(aug_edge_weight)
                        view_loss.backward()
                        torch.nn.utils.clip_grad_norm_(view_learner.parameters(), 1)
                        view_optimizer.step()

                        """STEP TWO - TRAIN THE MAIN MODEL"""
                        model.train()
                        model.zero_grad()
                        view_learner.eval()

                        out_score_logits, out_edge_feat, _, _ = model(data)
                        out = torch.sigmoid(out_score_logits)

                        # learn the edge weight (augmentation policy)
                        weight_logits = view_learner(data, device)

                        # gumbel softmax
                        # temperature = 1.0
                        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
                        eps = (bias - (1 - bias)) * torch.rand(weight_logits.size()) + (1 - bias)
                        gate_inputs = torch.log(eps) - torch.log(1 - eps)
                        gate_inputs = gate_inputs.to(device)
                        gate_inputs = (gate_inputs + weight_logits) / args.temperature
                        aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

                        # factual prediction
                        out_score_f_logits, out_edge_feat_f, _, _ = model(data, edge_weight=aug_edge_weight)
                        out_f = torch.sigmoid(out_score_f_logits)

                        # counterfactual prediction
                        edge_dropout_prob = 1 - aug_edge_weight
                        out_score_cf_logits, out_edge_feat_cf, _, _ = model(data, edge_weight=edge_dropout_prob)
                        out_cf = torch.sigmoid(out_score_cf_logits)

                        # factual loss
                        coef = out.detach().clone()
                        coef[out >= 0.5] = 1
                        coef[out < 0.5] = -1
                        loss_f = torch.mean(torch.clamp(torch.add(coef * (0 - out_score_f_logits), args.gamma), min=0))

                        # counter factual loss
                        coef = out.detach().clone()
                        coef[out >= 0.5] = -1
                        coef[out < 0.5] = 1
                        loss_cf = torch.mean(
                            torch.clamp(torch.add(coef * (0 - out_score_cf_logits), args.gamma), min=0))

                        # factual and counterfactual view loss
                        loss = args.view_alpha * loss_f + (1 - args.view_alpha) * loss_cf

                        model_loss = criterion(out[train_idx], data.y[train_idx]) + args.model_lambda * loss
                        model_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        model_optimizer.step()

                        # Test accuracy on the test set
                        # Load the best model based on validation performance
                if params is not None:
                    model.load_state_dict(params)
                test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g = GNN_evaluator(model=model, X=data.x,
                                                                                     hyperedge_index=data.edge_index,
                                                                                     Y=data.y,
                                                                                     test_idx=test_idx, data=data,
                                                                                     eval_func=eval_function,
                                                                                     method=args.method,
                                                                                     epoch=args.epochs,
                                                                                     args=args)

                print(f"Best validation accuracy: {valid_score:.4f} at epoch {best_ep}")
                print(
                    f"Test metrics - Accuracy: {test_acc_g:.4f}, AUC: {test_auc_g:.4f}, AUPR: {test_aupr_g:.4f}, F1 Macro: {test_f1_macro_g:.4f}")

                valid_auc, test_auc = float(valid_score), float(test_auc_g)
                results.append(test_auc)
        print(" Avg. Perf: {0} / Std. Perf: {1}".format(np.mean(results), np.std(results)))
    print('All done! Exit python code')
    quit()




