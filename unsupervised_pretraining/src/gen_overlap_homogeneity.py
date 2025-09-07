#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
============================================================
Hypergraph Preprocessing: Overlappness & Homogeneity
============================================================
This script builds basic statistics for a hypergraph dataset and saves them
for downstream models. It:
  • Loads hyperedges and (optional) node embeddings for <dataset_name>
  • Computes node "overlappness" (redundancy/duplication proxy)
  • Computes hyperedge "homogeneity" (average pairwise co-occurrence)
  • Persists tensors to disk for later training

------------------------------------------------------------
Inputs (CLI args)
  --data_path      Path to the dataset folder containing raw files
                   required files inside:
                     hyperedges-<dataset_name>.txt
                     node-embeddings-<dataset_name>            (text; first line: "<N> <D>")
  --dataset_name   Dataset identifier (e.g., mimic4)
  --raw_path       (Unused here; kept for interface compatibility)
  --num_node       Total number of nodes (int)

Derived paths (under --data_path)
  hyperedges-<dataset>.txt          # one line per hyperedge, comma-separated node ids
  node-embeddings-<dataset>         # text embeddings; row 0: "<N> <D>", then "<id> <d1> ... <dD>"

------------------------------------------------------------
Key Components
  • HyperDataset
      - load_graph():    reads 'hyperedges-*.txt' → dict[edge_id] = [node_ids]
      - load_features(): reads 'node-embeddings-*' → np.ndarray [num_node, dim]
  • cal_overlappness():            computes node-level redundancy score
  • cal_degree_of_each_pair():     builds node×node co-occurrence matrix
  • cal_homogeneity_hyperedge():   averages co-occurrence within each hyperedge
  • main(): end-to-end pipeline and serialization

------------------------------------------------------------
Usage
  python gen_overlap_homogeneity.py \
      --data_path /path/to/dataset/folder \
      --dataset_name dataset_name \
      --num_node num_node

"""



import argparse
import random
import scipy.sparse as sp
import torch_sparse
import torch
import json
import os
from pathlib import Path
import numpy as np

class HyperDataset:
    def __init__(self, data_path,dname):
        self.PATH = data_path
        self.dname = dname
        self.G = {}
        self.FEATURES = []
        self.NODE = []
        self.NODE_SIZE = None
        self.EDGE_SIZE = None
        # self.LABELS = []  #promote and mimic dont have node label
        # self.ID2LABEL = {}
        self.EDGE_LABELS = []

        # Load Hypher Graph
        self.load_graph()

        # Load Node labels
        # self.load_labels()

        # Load Node features
        self.load_features()

        # Load Edge Labels
        # self.load_edge_labels()

    def load_graph(self):
        """Load hypergraph structure from 'hyperedges-{dataset}.txt'."""
        file_path = os.path.join(self.PATH, f"hyperedges-{self.dname}.txt")
        with open(os.path.join(self.PATH, file_path), "r", encoding="utf-8") as f:
            idx = 0
            for line in f.readlines():
                edge_item = list(map(int, line.split(",")))
                self.G[idx] = edge_item
                idx += 1
                self.NODE.extend([node for node in edge_item])
        # Deduplicate node list
        self.NODE = list(set(self.NODE))
        self.NODE_SIZE = len(self.NODE)
        self.EDGE_SIZE = len(self.G)

    def load_labels(self):
        """Load node embeddings from 'node-embeddings-{dataset}'."""
        idx = 0
        with open(os.path.join(self.PATH, f"columns-{self.dname}.json")) as file:
            diagnosis_nodes = json.load(file)
            for i, (k, v) in enumerate(diagnosis_nodes.items()):
                self.LABELS.append(0)
                self.ID2LABEL[i + idx] = k


    def load_features(self):
        """Parse text embeddings file into numpy array [num_nodes, dim]."""
        file_path = os.path.join(self.PATH, f"node-embeddings-{self.dname}")
        num_node = max(self.NODE) + 1
        self.num_node = num_node

        try:
            # Parse the file and extract node features
            features = self.parse_node_embeddings(file_path, num_node)
            self.FEATURES = features

            # Ensure the feature matrix has correct dimensions
            assert len(self.FEATURES) == num_node, "Feature matrix row count mismatch with num_node"

        except Exception as e:
            raise RuntimeError(f"Error while loading features from {file_path}: {e}")

    @staticmethod
    def parse_node_embeddings(file_path, num_node):
        """
        Parse the node embeddings file.

        :param file_path: Path to the embeddings file.
        :param num_node: Total number of nodes.
        :return: A numpy array of node features.
        """
        try:
            with open(file_path, "r", encoding="utf8") as f:
                # Read the first line to determine the embedding dimension
                first_line = f.readline().strip()
                n_node, embedding_dim = map(int, first_line.split(" "))

                # Initialize feature matrix with random values
                features = np.random.rand(num_node, embedding_dim)

                # Parse the remaining lines
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        # Parse node ID and feature values
                        values = list(map(float, line.split(" ")))
                        node_id = int(values[0])  # First value is the node ID
                        features[node_id] = np.array(values[1:])  # Remaining values are the feature vector
                    except (ValueError, IndexError) as e:
                        print(f"Skipping malformed line: {line}")
                        continue

                return features
        except FileNotFoundError:
            raise FileNotFoundError(f"Embeddings file not found at {file_path}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while parsing {file_path}: {e}")

    def load_overlappness(self):
        """Load precomputed overlappness tensor."""
        f_path_overlappness = os.path.join(self.PATH, 'overlappness')
        overlappness = torch.load(f_path_overlappness)
        return overlappness

    def load_homogeneity(self):
        """Load precomputed homogeneity tensor."""
        f_homogeneity = os.path.join(self.PATH, 'homogeneity')
        homogeneity = torch.load(f_homogeneity)
        return homogeneity

    def load_edge_labels(self):
        """Load optional edge labels (if available)."""
        file_path = os.path.join(self.PATH, f"edge-labels-{self.dname}.txt")
        with open(os.path.join(self.PATH, file_path), "r", encoding="utf-8") as f:
            for line in f.readlines():
                label = list(map(int, line.split(",")))
                self.EDGE_LABELS.append(label)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def cal_overlappness(nlist, G:dict,num_node:int):
    '''
    :param nlist: node list
    :param G: hyperedges
    :return: tensor[nlist]

    Define: E{v9} = {e1, e4} contains 6 nodes:
    v1, v2, v6, v7, v8, v9, while E{v4} = {e2, e3} contains 5 nodes:
    vj (3 ≤ j ≤ 7). In this case, masking node v9 results in higher
    information loss than masking node v4. Second, the number of
    nodes contained in E{v6} = {e3, e4} is 6, which is the same
    as for E{v9}. However, v6 is less important than v9 because (i)
    E{v6} = E{v7}, meaning that e3 and e4 can still be connected via
    v7 even if v6 is masked; and (ii) E{v9}∩E{v1} = {e1} 6= E{v9},
    and E{v9} ∩ E{v8} = {e4} 6= E{v9}, meaning that e1 and e4
    cannot be connected if v9 is masked.

    e.g:
    nlist = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    G = {
        "e1": [1, 2, 9],
        "e2": [3, 4],
        "e3": [4, 5, 6, 7],
        "e4": [6, 7, 8, 9]
    }
    '''

    edgedict = {i: [] for i in nlist}

    for hyperedge in G.values():
        for node in hyperedge:
            if node in edgedict:  # Ensure the node is in nlist
                edgedict[node].extend(hyperedge)

    overlappness = torch.zeros(num_node)
    for E in edgedict.keys():
        subgraph = edgedict[E]
        subgraph_set = set(subgraph)
        up = len(subgraph)
        down = len(subgraph_set)

        if up != 0 and down != 0:
            overlappness[E] = up / down

    return overlappness


def cal_degree_of_each_pair(nlist,G:dict,num_node:int):
    """
    Compute co-occurrence matrix of nodes.
    degree_matrix[i, j] = #hyperedges containing both node i and j.
    """
    # degree_matrix = torch.zeros(len(nlist),len(nlist))
    degree_matrix = torch.zeros(num_node, num_node)
    for hyperedge in G.keys():
        e = G[hyperedge]
        for node_i in e:
            for node_j in e:
                degree_matrix[node_i-1,node_j-1] = degree_matrix[node_i-1,node_j-1] + 1 
    return degree_matrix


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def cal_homogeneity_hyperedge(nlist,G:dict,degree_matrix:torch.Tensor, num_node:int):
    """
        Compute homogeneity score for each hyperedge:
          - Average pairwise co-occurrence of its nodes
          - Apply sigmoid normalization
    """
    homogeneity = torch.ones(len(G))  # Initialize homogeneity for hyperedges with default value 1

    for i, hyperedge in enumerate(G.keys()):
        e = G[hyperedge]
        if len(e) > 1:
            homo = 0
            for node_i in e:
                for node_j in e:
                    if node_i != node_j:
                        homo += degree_matrix[node_i - 1, node_j - 1].item()
            homo = homo / (len(e) * (len(e) - 1))
            homogeneity[i] = sigmoid(homo)  # Update homogeneity for the hyperedges present

    return homogeneity


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))#numpy转成torch

    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def initialise(X, G):
    G = G.copy()
    N = X.shape[0]
    M = 8000
    indptr, indices, data = [0], [], []
    for i, (e, vs) in enumerate(G.items()):
        if i >= M:
            break
        indices += vs
        data += [1] * len(vs)
        indptr.append(len(indices))

    H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr()  # V x E
    (V, E), value = torch_sparse.from_scipy(H)
    H = sparse_mx_to_torch_sparse_tensor(H).to_dense()
    H = H.bool()
    return H, V, E


def normalise(M):
    d = np.array(M.sum(1))
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)  # D inverse i.e. D^{-1}

    return DI.dot(M)



def main(args):
    # Resolve paths
    DATA_PATH = args.data_path
    dname = args.dataset_name
    num_node = args.num_node
    p2raw = args.raw_path

    # Load dataset
    dataset = HyperDataset(DATA_PATH, dname)
    len_Y = dataset.EDGE_SIZE
    nlist = np.arange(len_Y)
    G = dataset.G  # Retrieve hypergraph structure

    # Overlappness
    f_path_overlappness = os.path.join(DATA_PATH, 'overlappness')
    overlappness = cal_overlappness(nlist, G, num_node)
    torch.save(overlappness, f_path_overlappness)

    # Homogeneity
    degree_matrix = cal_degree_of_each_pair(nlist, G, dataset.num_node)
    homogeneity = cal_homogeneity_hyperedge(nlist, G, degree_matrix, num_node)
    f_path_homogeneity = os.path.join(DATA_PATH, 'homogeneity')
    torch.save(homogeneity, f_path_homogeneity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hypergraph dataset preprocessing")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset directory (e.g., /local/.../mimic4)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (e.g., mimic4)"
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        default="path_to_raw_data_Dir",
        help="Path to raw data root directory"
    )
    parser.add_argument(
        "--num_node",
        type=int,
        required=True,
        help="Number of nodes in the hypergraph"
    )

    args = parser.parse_args()
    main(args)