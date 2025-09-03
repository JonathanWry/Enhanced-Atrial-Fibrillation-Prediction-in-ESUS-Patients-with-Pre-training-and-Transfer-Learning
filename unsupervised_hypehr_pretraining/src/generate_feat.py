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