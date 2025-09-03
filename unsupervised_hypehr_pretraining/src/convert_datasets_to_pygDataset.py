import torch
import pickle
import os
import ipdb

import os.path as osp
import numpy as np
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_sparse import coalesce


def load_dataset(path='../data/raw_data/', dataset='your_dataset',
                 node_feature_path="../data/mimic3/node-embeddings-your_dataset",
                 num_node=7423, upperFolder=None):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - add gaussian noise with sigma = nosie, mean = one hot coded label.

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    print(f'Loading hypergraph dataset from: {dataset}')
    print(f'node_feature_path:{node_feature_path}')

    # first load edge labels
    if upperFolder is None:
        label_file_path = osp.join(path, dataset, f'edge-labels-{dataset}.txt')
        homogeneity_path = osp.join(path, dataset, 'homogeneity')
        overlappness_path = osp.join(path, dataset, 'overlappness')
        p2hyperedge_list = osp.join(path, dataset, f'hyperedges-{dataset}.txt')
    else:
        label_file_path = osp.join(path, upperFolder,dataset, f'edge-labels-{dataset}.txt')
        homogeneity_path = osp.join(path, upperFolder, dataset, 'homogeneity')
        overlappness_path = osp.join(path, upperFolder, dataset, 'overlappness')
        p2hyperedge_list = osp.join(path, upperFolder,dataset, f'hyperedges-{dataset}.txt')
    if osp.isfile(label_file_path):
        df_labels = pd.read_csv(label_file_path, sep=',', header=None, encoding='utf-8')
        num_edges = df_labels.shape[0]
    else:
        # Generate random placeholder labels
        labels = torch.FloatTensor(num_node, 1).uniform_(0, 1)

    # then create node features.
    with open(node_feature_path, 'r') as f:
        line = f.readline().strip()
        # print(line)
        try:
            n_node, embedding_dim = map(int, line.split(" "))
        except ValueError as e:
            raise RuntimeError(f"Failed to parse the first line of {node_feature_path}: {line}. Error: {e}")
        print("num_node here:", num_node)
        features = np.random.rand(num_node, int(embedding_dim))
        for lines in f:
            lines = lines.strip()  # Remove leading/trailing whitespace
            if not lines:  # Skip empty lines
                continue
            try:
                values = list(map(float, lines.split(" ")))
                features[int(values[0])] = np.array(values[1:])
            except ValueError as e:
                print(f"Skipping malformed line: {lines.strip()}")
                continue

    num_nodes = features.shape[0]

    print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')



    if osp.exists(homogeneity_path):
        homo = torch.load(homogeneity_path)
        print(f"Loaded homogeneity data from {homogeneity_path}")
    else:
        homo = None  # Assign a default value if the file doesn't exist
        print(f"Homogeneity file not found at {homogeneity_path}, assigning None.")

        # Load overlappness if the file exists
    if osp.exists(overlappness_path):
        overlap = torch.load(overlappness_path)
        print(f"Loaded overlappness data from {overlappness_path}")
    else:
        overlap = None  # Assign a default value if the file doesn't exist
        print(f"Overlappness file not found at {overlappness_path}, assigning None.")

    features = torch.FloatTensor(features)
    # labels = torch.FloatTensor(labels)


    node_list = []
    he_list = []
    he_id = num_nodes

    with open(p2hyperedge_list, 'r') as f:
        num_hyperedges = sum(1 for line in f)

    with open(p2hyperedge_list, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            cur_set = line.split(',')
            cur_set = [int(x) for x in cur_set]

            node_list += cur_set
            he_list += [he_id] * len(cur_set)
            he_id += 1
    # shift node_idx to start with 0.
    node_idx_min = np.min(node_list)
    node_list = [x - node_idx_min for x in node_list]

    edge_index = [node_list + he_list,
                  he_list + node_list]

    edge_index = torch.LongTensor(edge_index)

    data = Data(x=features,
                edge_index=edge_index,
                y=labels,
                overlap=overlap,
                homo=homo
                )
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates.
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index,
                                               None,
                                               total_num_node_id_he_id,
                                               total_num_node_id_he_id)

    n_x = num_nodes
    data.n_x = n_x
    data.num_hyperedges = he_id - num_nodes
    data.n_label=num_hyperedges

    return data


def save_data_to_pickle(data, p2root='../data/', file_name=None):
    """
    Save arbitrary Python object to a pickle file.

    Args:
        data (Any): Python object to be serialized.
        p2root (str): Root directory where file will be stored. Defaults to '../data/'.
        file_name (str, optional): Custom file name. If None, a default name is used.

    Returns:
        str: Full path to the saved pickle file.

    Notes:
        - Default file name is "Hypergraph_star_expansion_dataset".
        - Creates the target directory if it does not exist.
    """
    surfix = 'star_expansion_dataset'
    if file_name is None:
        tmp_data_name = '_'.join(['Hypergraph', surfix])
    else:
        tmp_data_name = file_name
    p2he_StarExpan = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2he_StarExpan, 'bw') as f:
        pickle.dump(data, f)
    return p2he_StarExpan


class dataset_Hypergraph(InMemoryDataset):
    """
        A PyTorch Geometric InMemoryDataset for hypergraph data.

        This dataset class handles loading raw hypergraph data, processing it,
        and saving it in a format suitable for PyTorch Geometric.

        Supported dataset names:
            - mimic3
            - cradle
            - promote
            - combine_icd3
            - combine_icd4
            - separate_* (custom datasets with 'separate_' prefix)

        Args:
            root (str): Root directory where the dataset will be stored.
            name (str): Dataset name.
            p2raw (str, optional): Path to the raw dataset.
            transform (callable, optional): Data transform.
            pre_transform (callable, optional): Pre-transform applied before saving.
            num_nodes (int): Number of nodes in the dataset.
        """
    def __init__(self, root='../data/pyg_data/hypergraph_dataset/', name=None,
                 p2raw=None, transform=None, pre_transform=None, num_nodes=7423):

        existing_dataset = ['mimic3', 'cradle', 'promote','combine_icd3','combine_icd4']
        if name not in existing_dataset and not (name.startswith('separate_')):
            raise ValueError(f'name of hypergraph dataset must be one of: {existing_dataset}')
        elif name.startswith('separate_'):
            parts = name.split('_')
            if len(parts) < 3:
                raise ValueError(f"Dataset name '{name}' is not formatted correctly for 'separate_' prefix.")
            self.name = parts[-1]  # Last part after the last underscore
            self.upperfolder = '_'.join(parts[:-1])
            self.myraw_dir = osp.join(root, self.upperfolder, self.name, 'raw')
            self.myprocessed_dir = osp.join(root, self.upperfolder, self.name, 'processed')
        else:
            self.name = name
            self.myraw_dir = osp.join(root, self.name, 'raw')
            self.myprocessed_dir = osp.join(root, self.name, 'processed')
        print(os.getcwd())
        print(p2raw)
        print(os.listdir(p2raw))
        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        self.num_nodes = num_nodes
        super(dataset_Hypergraph, self).__init__(osp.join(root, name), transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.n_label = self.data.n_label

    # @property
    # def raw_dir(self):
    #     return osp.join(self.root, self.name, 'raw')

    # @property
    # def processed_dir(self):
    #     return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        file_names = ['data.pt']
        return file_names

    @property
    def num_features(self):
        return self.data.num_node_features

    def download(self):
        for name in self.raw_file_names:
            p2f = osp.join(self.myraw_dir, name)
            if not osp.isfile(p2f):
                # file not exist, so we create it and save it there.
                print(p2f)
                print(self.p2raw)
                print(self.name)

                if self.name in ['mimic3']:
                    tmp_data = load_dataset(path=self.p2raw,
                                            dataset=self.name,
                                            node_feature_path="../data/raw_data/mimic3/node-embeddings-mimic3",
                                            num_node=self.num_nodes)

                elif self.name in ['cradle']:
                    tmp_data = load_dataset(path=self.p2raw,
                                            dataset=self.name,
                                            node_feature_path="../data/raw_data/cradle/node-embeddings-cradle",
                                            num_node=self.num_nodes)
                elif self.upperfolder.startswith('separate_'):
                    print("node feature path:::")
                    print(f"../data/raw_data/{self.upperfolder}/{self.name}/node-embeddings-{self.name}")
                    tmp_data = load_dataset(path=self.p2raw,
                                            dataset=self.name,
                                            node_feature_path=f"../data/raw_data/{self.upperfolder}/{self.name}/node-embeddings-{self.name}",
                                            num_node=self.num_nodes,
                                            upperFolder=self.upperfolder)                     
                elif self.name in ['promote']:
                    tmp_data = load_dataset(path=self.p2raw,
                                            dataset=self.name,
                                            node_feature_path="../data/raw_data/promote/node-embeddings-promote",
                                            num_node=self.num_nodes)
                elif self.name in ['combine_icd3']:
                    tmp_data = load_dataset(path=self.p2raw,
                                            dataset=self.name,
                                            node_feature_path="../data/raw_data/combine_icd3/node-embeddings-combine_icd3",
                                            num_node=self.num_nodes)
                elif self.name in ['combine_icd4']:
                    tmp_data = load_dataset(path=self.p2raw,
                                            dataset=self.name,
                                            node_feature_path="../data/raw_data/combine_icd4/node-embeddings-combine_icd4",
                                            num_node=self.num_nodes)    
                

                _ = save_data_to_pickle(tmp_data,
                                        p2root=self.myraw_dir,
                                        file_name=self.raw_file_names[0])
            else:
                # file exists already. Do nothing.
                pass

    def process(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)
