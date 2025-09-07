#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
============================================================
Random Walks + Word2Vec: Node Embedding Generator
============================================================
Builds node embeddings from a hyperedge file by:
  • Converting each hyperedge (comma-separated node list) into a simple graph
  • Running uniform random walks on the graph
  • Training a Word2Vec model on the walk sequences
  • Saving embeddings in text format compatible with downstream loaders

------------------------------------------------------------
Pipeline
  1) Read hyperedges-*.txt  → undirected graph (cliques per hyperedge)
  2) Perform N random walks per node (length L)
  3) Train Word2Vec (skip-gram) on walk sequences
  4) Write "<num_nodes> <dim>" header + "<node> <dim_values...>" per line

------------------------------------------------------------
Inputs (CLI)
  --data_dir       Directory containing the hyperedge file
  --input_file     Hyperedge file name (e.g., hyperedges-mimic4.txt)
  --output_file    Output embeddings file (text)
  --seed           RNG seed (default: 0)
  --num_walks      Walks per node (default: 10)
  --walk_length    Steps per walk (default: 40)
  --vector_size    Embedding dimension (default: 128)
  --window         Word2Vec context window (default: 5)
  --epochs         Word2Vec epochs (default: 10)
  --workers        Word2Vec worker threads (default: 4)

------------------------------------------------------------
Usage:
    python generate_feat.py \
      --data_dir path_to_dataset \
      --input_file path_to_hyperedges.txt \
      --output_file path_to_node-embeddings-dataset.txt \
      --seed 0 \
      --num_walks 10 \
      --walk_length 40 \
      --vector_size 128 \
      --window 5 \
      --epochs 10
"""

import argparse

import networkx as nx
from gensim.models import Word2Vec
import random
import os


def seed_everything(seed=0):
    random.seed(seed)  

def read_and_preprocess(file_path):
    # Read hyperedge file and build an undirected graph
    with open(file_path, 'r') as file:
        content = file.read()
    edges = [line.split(',') for line in content.strip().split('\n')]
    G = nx.Graph()
    for edge in edges:
        for i in range(len(edge)):
            for j in range(i + 1, len(edge)):
                G.add_edge(edge[i], edge[j])
    return G

def perform_random_walks(graph, num_walks=10, walk_length=40):
    # Perform uniform random walks starting from each node
    walks = []
    for node in list(graph.nodes):
        for _ in range(num_walks):
            walk = [node]
            while len(walk) < walk_length:
                cur = walk[-1]
                cur_neighbors = list(graph.neighbors(cur))
                if len(cur_neighbors) > 0:
                    walk.append(random.choice(cur_neighbors))
                else:
                    break
            walks.append(list(map(str, walk)))
    return walks

def generate_embeddings(walks, vector_size=128, window=5, min_count=0, sg=1, workers=4, epochs=10):
    # Train Word2Vec model on the generated random walks
    model = Word2Vec(sentences=walks, vector_size=vector_size, window=window, min_count=min_count, sg=sg, workers=workers, epochs=epochs)
    return model

def save_embeddings_to_txt(model, file_path):
    vocab = list(model.wv.index_to_key)
    embeddings = model.wv[vocab]
    with open(file_path, 'w') as f:
        f.write(f"{len(vocab)} {model.vector_size}\n")
        for node, embedding in zip(vocab, embeddings):
            f.write(f"{node} {' '.join(map(str, embedding))}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate node embeddings with random walks + Word2Vec")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the hyperedge file")
    parser.add_argument("--input_file", type=str, required=True, help="Input hyperedge file (e.g., hyperedges-mimic4.txt)")
    parser.add_argument("--output_file", type=str, required=True, help="Output embeddings file path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_walks", type=int, default=10, help="Number of walks per node")
    parser.add_argument("--walk_length", type=int, default=40, help="Length of each random walk")
    parser.add_argument("--vector_size", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec context window size")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for Word2Vec")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")

    args = parser.parse_args()

    os.chdir(args.data_dir)
    seed_everything(seed=args.seed)

    G = read_and_preprocess(args.input_file)
    walks = perform_random_walks(G, num_walks=args.num_walks, walk_length=args.walk_length)
    model = generate_embeddings(
        walks,
        vector_size=args.vector_size,
        window=args.window,
        epochs=args.epochs,
        workers=args.workers,
    )
    save_embeddings_to_txt(model, args.output_file)


if __name__ == "__main__":
    main()