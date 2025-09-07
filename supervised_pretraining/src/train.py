#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
======================================================================================
Main Training Script: Supervised Pretraining & Formal Training for Hypergraph Representation
======================================================================================


Purpose
-------
Implements the experimental pipeline to train hypergraph-based models with
optional “vanilla” training (attention-weight supervision) and utilities for
view generation and evaluation.

  • parse_method(args, data)
      Builds the model (currently SetGNN) given CLI args and data.

  • seed_everything(seed)
      Sets seeds across Python, NumPy, and PyTorch for reproducibility.

  • evaluate(model, data, split_idx, eval_func, epoch, method, dname, args)
      Evaluates the model on validation/test splits and returns metrics.
      Supports reading attention weights for analysis when --vanilla is set.

  • eval_pretraining(y_true, y_pred, ...)
      Computes ACC, ROC-AUC, AUPR, and macro-F1 for multilabel edge prediction.

  • Main Execution (__main__)
      - Loads and prepares dataset (including optional self-loops & normalization)
      - Builds SetGNN and ViewLearner
      - Trains (vanilla mode supported) and logs metrics
      - Saves model; optionally computes embeddings on a second dataset (ESUS)

Dependencies
------------
torch, torch_geometric, torch_scatter, sklearn, tqdm, numpy
Custom modules: layers, models, preprocessing, convert_datasets_to_pygDataset

Usage
------
python -u main_train.py \
  --dname pre-training --epochs 600 --cuda 1 --num_labels 1 --num_nodes 12725 \
  --num_labeled_data all --All_num_layers 3 --feature_dim 128 --heads 4 \
  --MLP_num_layers 2 --MLP_hidden 32 --model_lambda 0.01 --vanilla \
  --train_prop 0.7 --valid_prop 0.1
"""

import os
import time
import torch
import pickle
import argparse

import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm, trange

from layers import *
from models import *
from preprocessing import *

from convert_datasets_to_pygDataset import dataset_Hypergraph
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score

from copy import deepcopy
import random
import time


def parse_method(args, data):
    model = None
    if args.dname in ['pre-training']:
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

@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, epoch, method, dname, args):
    """
    Run evaluation on validation and test splits.

    Args:
        model: trained/partially trained torch.nn.Module.
        data: PyG Data; must contain x, edge_index, y, and norm (if required).
        split_idx (dict): {'train','valid','test'} tensor indices.
        eval_func (callable): function(y_true, y_pred, ...) -> metrics.
        epoch (int): current epoch for logging.
        method (str): model/method name for logging context.
        dname (str): dataset name.
        args: argparse.Namespace with flags like threshold, vanilla.

    Returns:
        Tuple of validation/test metrics (ACC, AUC, AUPR, F1 macro) for
        different regimes (placeholders for *_gf and *_gcf currently 0).
    """

    valid_acc_gf = valid_auc_gf = valid_aupr_gf = valid_f1_macro_gf = \
    test_acc_gf = test_auc_gf = test_aupr_gf = test_f1_macro_gf = \
    valid_acc_gcf = valid_auc_gcf = valid_aupr_gcf = valid_f1_macro_gcf = \
    test_acc_gcf = test_auc_gcf = test_aupr_gcf = test_f1_macro_gcf = 0

    model.eval()

    # Forward on the original graph
    out_score_g_logits, edge_feat, node_feat, weight_tuple = model(data)
    out_g = torch.sigmoid(out_score_g_logits)
    # Compute metrics via provided evaluator
    valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g = eval_func(
        data.y[split_idx['valid']], out_g[split_idx['valid']],
        epoch, method, dname, args, mode='dev_g', threshold=args.threshold)
    test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g = eval_func(data.y[split_idx['test']],
                                                                     out_g[split_idx['test']],
                                                                     epoch, method, dname, args,
                                                                     mode='test_g',
                                                                     threshold=args.threshold)
    # Optional: inspect attention/edge weights in vanilla mode
    if args.vanilla:
        edge_index = weight_tuple[0]
        edge_weight = weight_tuple[1].reshape(-1)
        # num_hyperedges = data.num_hyperedges[0]
        num_hyperedges = data.num_hyperedges
        # if epoch == args.epochs - 1:
            # get_subset_ranking(edge_weight, edge_index, num_hyperedges, args)

    return valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g, \
           test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g, \
           valid_acc_gf, valid_auc_gf, valid_aupr_gf, valid_f1_macro_gf, \
           test_acc_gf, test_auc_gf, test_aupr_gf, test_f1_macro_gf, \
           valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf, \
           test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf


def eval_pretraining(y_true, y_pred, epoch, method, dname, args, mode='dev', threshold=0.5):
    """
    Compute standard multilabel classification metrics at a fixed threshold.

    Args:
        y_true (Tensor): ground-truth labels.
        y_pred (Tensor): predicted probabilities (pre-threshold).
        epoch, method, dname, args: passthrough for logging compatibility.
        mode (str): label for logging (e.g., 'dev' or 'test').
        threshold (float): decision threshold for binarization.

    Returns:
        (accuracy, roc_auc, aupr, f1_macro)
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


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.7)
    parser.add_argument('--valid_prop', type=float, default=0.1)
    parser.add_argument('--dname', default='pre-training')
    parser.add_argument('--method', default='AllSetTransformer')
    parser.add_argument('--epochs', default=0, type=int)
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-3, type=float)
    parser.add_argument('--view_lr', default=1e-2, type=float)
    parser.add_argument('--view_wd', default=1e-3, type=float)
    # How many layers of full NLConvs
    parser.add_argument('--All_num_layers', default=3, type=int)
    parser.add_argument('--MLP_num_layers', default=2,
                        type=int)  # How many layers of encoder
    parser.add_argument('--MLP_hidden', default=8,
                        type=int)  # Encoder hidden units
    parser.add_argument('--Classifier_num_layers', default=3,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=64,
                        type=int)  # Decoder hidden units
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    # ['all_one','deg_half_sym']
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_labels', default=1, type=int)  # set the default for now
    parser.add_argument('--num_nodes', default=2639, type=int)  # 7423 for mimic and 12725 for pre-training
    # 'all' means all samples have labels, otherwise it indicates the first [num_labeled_data] rows that have the labels
    parser.add_argument('--num_labeled_data', default='all', type=str)
    parser.add_argument('--feature_dim', default=128, type=int)  # feature dim of learnable node feat
    parser.add_argument('--LearnFeat', action='store_true')
    # whether the he contain self node or not
    parser.add_argument('--PMA', action='store_true')
    #     Args for Attentions
    parser.add_argument('--heads', default=1, type=int)  # Placeholder
    parser.add_argument('--output_ ', default=1, type=int)  # Placeholder

    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--view_alpha', type=float, default=0.5)
    parser.add_argument('--view_lambda', type=float, default=5)
    parser.add_argument('--model_lambda', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=1)  # 0.5 | 5; temperature for gumbel softmax

    parser.add_argument('--vanilla', action='store_true')
    parser.add_argument('--remain_percentage', default=0.3, type=float)
    parser.add_argument('--rand_seed', default=0, type=int)
    parser.add_argument('--random_split', action='store_true', default=False)
    parser.add_argument('--MLP_tuning', action='store_true', default=False)
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.set_defaults(PMA=True)
    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(LearnFeat=False)

    args = parser.parse_args()
    
    seed_everything(args.rand_seed) 

    existing_dataset = ['mimic3', 'pre-training', 'promote']

    synthetic_list = ['mimic3', 'pre-training', 'promote']

    dname = args.dname
    p2raw = '../data/raw_data/'
    dataset = dataset_Hypergraph(name=dname, root='../data/pyg_data/hypergraph_dataset/',
                                 p2raw=p2raw, num_nodes=args.num_nodes)
    data = dataset.data
    args.num_features = dataset.num_features
    if args.dname in ['pre-training']:
        # Shift the y label to start with 0
        data.y = data.y - data.y.min()
    if not hasattr(data, 'n_x'):
        data.n_x = torch.tensor([data.x.shape[0]])
    if not hasattr(data, 'num_hyperedges'):
        # note that we assume the he_id is consecutive.
        data.num_hyperedges = torch.tensor(
            [data.edge_index[0].max() - data.n_x[0] + 1])
    # Preprocess incidence edges + norms
    if args.method == 'AllSetTransformer':
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        # Build incidence normalization weights (all ones by default)
        data = norm_contruction(data, option=args.normtype)

    if args.load_model:
        model = torch.load('model.pt')
    else:
        model = parse_method(args, data)
    

    view_learner = ViewLearner(parse_method(args, data), args.MLP_hidden)
    # put things to device
    if args.cuda != '-1':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    
    split_idx = rand_train_test_idx(data.y, train_prop=args.train_prop, valid_prop=args.valid_prop, rand_seed=args.rand_seed, random_split=args.random_split)
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    model, view_learner, data = model.to(device), view_learner.to(device), data.to(device)

    criterion = nn.BCELoss()

    model.train()
    model.reset_parameters()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr, weight_decay=args.view_wd)

    with open(f'../data/raw_data/{args.dname}/hyperedges-{args.dname}.txt', 'r') as f:
        total_edges = []
        maxlen = 0
        for lines in f:
            line = lines.strip().split(',')
            line = list(map(int, line))
            if len(line) > maxlen:
                maxlen = len(line)
            total_edges.append(line)
        total_edges_padded = []
        for edge in total_edges:
            total_edges_padded.append(edge + [-1] * (maxlen - len(edge)))

    if args.num_labeled_data != 'all':
        N = int(args.num_labeled_data)  # the first x visits have labels
    elif args.num_labeled_data == 'all':
        N = len(total_edges_padded)  # all the samples in pre-training have labels
    train_num = int(N * args.train_prop)
    valid_num = int(N * args.valid_prop)
    train_input = torch.LongTensor(total_edges_padded[:train_num]).to(device)
    dev_input = torch.LongTensor(total_edges_padded[train_num:train_num + valid_num]).to(device)
    test_input = torch.LongTensor(total_edges_padded[train_num + valid_num:N]).to(device)

    edge_id_dict = None
    with torch.autograd.set_detect_anomaly(True):
        for epoch in trange(args.epochs):
            if args.vanilla:  # VANILLA - Use attention weight to get an important set for each encounter
                model.train()
                model.zero_grad()

                out_score_logits, _, _, weight_tuple = model(data)
                out = torch.sigmoid(out_score_logits)

                model_loss = criterion(out[train_idx], data.y[train_idx]) + args.view_lambda * torch.mean(
                    weight_tuple[1].reshape(-1))
                model_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                model_optimizer.step()

            if dname in ['pre-training']:
                eval_function = eval_pretraining
            valid_acc_g, valid_auc_g, valid_aupr_g, valid_f1_macro_g, \
            test_acc_g, test_auc_g, test_aupr_g, test_f1_macro_g, \
            valid_acc_gf, valid_auc_gf, valid_aupr_gf, valid_f1_macro_gf, \
            test_acc_gf, test_auc_gf, test_aupr_gf, test_f1_macro_gf, \
            valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf, \
            test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf = \
                evaluate(model, data, split_idx, eval_function, epoch, args.method, args.dname,
                         args)

            fname_dev = ''
            fname_test = ''
            vanilla = ""
            if args.vanilla: vanilla = "_vanilla"
            if dname == 'pre-training':
                fname_dev = f'outputs/pre-training_dev_{args.method}{vanilla}.txt'
                fname_test = f'outputs/pre-training_test_{args.method}{vanilla}.txt'
            # dev set
            with open(fname_dev, 'a+', encoding='utf-8') as f:
                f.write(
                    'Epoch: {}, Threshold: {:.2f}, lr: {:.2e}, wd: {:.2e}, view_lr: {:.2e}, view_wd: {:.2e}, '
                    'view_alpha:{:.2f}, view_lambda:{:.3f}, model_lambda:{:.3f}, gamma:{:.2f}, ACC_G: {:.5f}, '
                    'AUC_G: {:.5f}, AUPR_G: {:.5f}, F1_MACRO_G: {:.5f}, ACC_Gf: {:.5f}, AUC_Gf: {:.5f}, AUPR_Gf: {:.5f}, F1_MACRO_Gf: {:.5f}, '
                    'ACC_Gcf: {:.5f}, AUC_Gcf: {:.5f}, AUPR_Gcf: {:.5f}, F1_MACRO_Gcf: {:.5f}\n '
                        .format(epoch + 1, args.threshold, args.lr, args.wd, args.view_lr, args.view_wd,
                                args.view_alpha, args.view_lambda, args.model_lambda, args.gamma, valid_acc_g,
                                valid_auc_g, valid_aupr_g, valid_f1_macro_g, valid_acc_gf, valid_auc_gf, valid_aupr_gf,
                                valid_f1_macro_gf,
                                valid_acc_gcf, valid_auc_gcf, valid_aupr_gcf, valid_f1_macro_gcf))
            # test set
            with open(fname_test, 'a+', encoding='utf-8') as f:
                f.write(
                    'Epoch: {}, Threshold: {:.2f}, lr: {:.2e}, wd: {:.2e}, view_lr: {:.2e}, view_wd: {:.2e}, '
                    'view_alpha:{:.2f}, view_lambda:{:.3f}, model_lambda:{:.3f}, gamma:{:.2f}, ACC_G: {:.5f}, '
                    'AUC_G: {:.5f}, AUPR_G: {:.5f}, F1_MACRO_G: {:.5f}, ACC_Gf: {:.5f}, AUC_Gf: {:.5f}, AUPR_Gf: {:.5f}, F1_MACRO_Gf: {:.5f}, '
                    'ACC_Gcf: {:.5f}, AUC_Gcf: {:.5f}, AUPR_Gcf: {:.5f}, F1_MACRO_Gcf: {:.5f}\n'
                        .format(epoch + 1, args.threshold, args.lr, args.wd, args.view_lr, args.view_wd,
                                args.view_alpha, args.view_lambda, args.model_lambda, args.gamma, test_acc_g,
                                test_auc_g, test_aupr_g, test_f1_macro_g, test_acc_gf, test_auc_gf, test_aupr_gf,
                                test_f1_macro_gf, test_acc_gcf, test_auc_gcf, test_aupr_gcf, test_f1_macro_gcf))
    model.eval()
    torch.save(model, 'model.pt')
    
    # prepare data for another dataset - AF
    dname = 'ESUS'
    p2raw = '../data/raw_data/'
    dataset = dataset_Hypergraph(name=dname, root='../data/pyg_data/hypergraph_dataset/',
                                 p2raw=p2raw, num_nodes=args.num_nodes)
    data = dataset.data
    args.num_features = dataset.num_features
    if args.dname in ['ESUS']:
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
   
    # get the generated embedding
    model.eval()
    embeddings = model.get_embedding(data.to(device))
    embeddings_cpu = embeddings.cpu().detach().numpy()
    np.save('embeddings.npy', embeddings_cpu)
    
    end_time = time.time()
    print(f'Total running time {end_time - start_time} seconds')
    print('All done! Exit python code')
    quit()
