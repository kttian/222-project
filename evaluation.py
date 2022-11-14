"""
This file performs evaluation on our datasets and heuristics
Given a list of datasets and a list of heuristics, computes the performance table

Performance metrics:
- accuracy: # edges predicted correctly / # missing edges in test set
"""
import argparse
import logging
import time

import networkx as nx
import matplotlib.pyplot as plt
from heuristics import *
from dataset import * 


def evaluation(G_test, G_train, heuristic_fn, log_edges_every=10000):
    """
    For every pair of vertices in train graph without an existing edge,
    predict whether a new edge will form there in the future
    problem: struggle with low # of actual future edges, also slow
    """
    logging.info(f"G_train: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")
    logging.info(f"G_test: {G_test.number_of_nodes()} nodes, {G_test.number_of_edges()} edges")
    edge_scores = []
    no_edge_scores = []
    num_edges_evaluated = 0
    time_tic = time.perf_counter()
    for u in G_train.nodes():
        for v in G_train.nodes():
            if u != v:
                score = heuristic_fn(G_train, u, v)
                if G_test.has_edge(u, v):
                    edge_scores.append(score)
                else:
                    no_edge_scores.append(score)

                # Time logging
                num_edges_evaluated += 1
                if num_edges_evaluated % log_edges_every == 0:
                    time_toc = time.perf_counter()
                    logging.info(f'Evaluated {log_edges_every} edges in {time_toc - time_tic:.2f} seconds')
                    time_tic = time.perf_counter()

    print("future edges", len(edge_scores))
    print("future non-edges", len(no_edge_scores))
    max_edge_score = np.nanmax(edge_scores)
    np.nan_to_num(edge_scores, copy=False, nan=max_edge_score+1)
    plt.hist(edge_scores)
    plt.hist(no_edge_scores)
    plt.show()

    '''
    alternate? for every edge in test graph that wasn't already in the train graph,
    see whether scores are predictive 
    '''

def score_vectorized(G_train, heuristic_fn_vec):
    '''
    Compute the score for every pair of nodes in the train graph
    To be used in scoring test edges 
    Returns:
        scores: NxN matrix of edge scores 
    '''
    # Obtain scores for all edges
    sorted_nodes = np.array(sorted(G_train.nodes()))
    scores = heuristic_fn_vec(G_train, nodelist=sorted_nodes)
    return scores 

def evaluation_vectorized(G_train, G_test, heuristic_fn):
    # Obtain new edges in test graph
    G_train_edges_set = set(G_train.edges())
    new_edges = set(G_test.edges()) - G_train_edges_set
    num_new_edges = len(new_edges)

    # obtain scores for all pairs of nodes in train graph
    sorted_nodes = np.array(sorted(G_train.nodes()))
    scores = score_vectorized(G_train, heuristic_fn)

    # Sort edges scores
    logging.info(f"Sorting scores")
    time_tic = time.perf_counter()
    sorted_scoring_edges_u, sorted_scoring_edges_v = np.unravel_index(np.argsort(scores, axis=None), shape=scores.shape)
    sorted_scoring_edges = zip(
        reversed(sorted_nodes[sorted_scoring_edges_u]),
        reversed(sorted_nodes[sorted_scoring_edges_v])
    )
    time_toc = time.perf_counter()
    logging.info(f"Sorted scores in {time_toc - time_tic:.2f} seconds")

    # Need to remove the edges that already exist in the train graph
    logging.info(f"Removing existing edges")
    time_tic = time.perf_counter()
    pred_edges = []
    for u, v in sorted_scoring_edges:
        if len(pred_edges) >= num_new_edges:
            break
        if (u, v) not in G_train_edges_set:
            pred_edges.append((u, v))
    time_toc = time.perf_counter()
    logging.info(f"Removed existing edges in {time_toc - time_tic:.2f} seconds")

    # Compute accuracy
    accuracy(pred_edges, new_edges)


def accuracy(pred_edges, new_edges):
    '''
    accuracy: # edges predicted correctly / # missing edges in test set
    '''
    # TODO: how to turn into a prediction?
    print(f"Number of correctly predicted edges: {len(set(pred_edges) & set(new_edges))}")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, required=True)
    # parser.add_argument('--model', type=str, required=True)
    # parser.add_argument('--output', type=str, required=True)
    # parser.add_argument('--data', type=str, required=True)
    # parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--batch_size', type=int, default=32)
    # args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(filename)s %(funcName)s(%(lineno)d) %(message)s',
    )

    G, metadata = load_dataset(with_date=True, small=-1)
    G = filter_graph(G)
    train_G = graph_subset(G, start_date='1994-01-01', end_date='1997-01-01')
    test_G = graph_subset(G, start_date='1997-01-01', end_date='2000-01-01')
    # train_G = filter_prolific_authors(train_G)
    # test_G = filter_prolific_authors(test_G)
    
    # evaluation(test_G, train_G, common_neighbors)
    evaluation_vectorized(train_G, test_G, common_neighbors_vectorized)
