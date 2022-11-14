"""
This file performs evaluation on our datasets and heuristics
Given a list of datasets and a list of heuristics, computes the performance table

Performance metrics:
- accuracy: # edges predicted correctly / # missing edges in test set
"""
import argparse
import logging
from pathlib import Path
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


def prediction_vectorized(G_train, heuristic_fn_vec, use_top_n_edges=None):
    """ Predicts edges in G_test using the heuristic function.

    :param G_train:
    :param G_test:
    :param heuristic_fn_vec:
    :param use_top_n_edges: Integer, top n scoring edges to use for prediction. If None, use number
        of new edges in G_test.
    :return:
    """
    assert(use_top_n_edges is not None)

    # obtain scores for all pairs of nodes in train graph
    sorted_nodes = np.array(sorted(G_train.nodes()))
    scores = score_vectorized(G_train, heuristic_fn_vec)

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

    # Obtain new edges in test graph
    G_train_edges_set = set(G_train.edges())

    # Need to remove the edges that already exist in the train graph
    logging.info(f"Removing existing edges")
    time_tic = time.perf_counter()
    pred_edges = []
    for u, v in sorted_scoring_edges:
        if len(pred_edges) >= use_top_n_edges:
            break
        if (u, v) not in G_train_edges_set:
            pred_edges.append((u, v))
    time_toc = time.perf_counter()
    logging.info(f"Removed existing edges in {time_toc - time_tic:.2f} seconds")

    return pred_edges


def evaluation(pred_edges, expected_edges, step_size=1000, project_dir=Path.cwd()):
    """ Evaluate the performance of the prediction.

    :param pred_edges:
    :param new_edges:
    :param step_size:
    :param project_dir:
    :return:
    """
    # Create result directory
    res_dir = project_dir / 'res'
    if not res_dir.exists():
        res_dir.mkdir(exist_ok=True)

    all_correct_pred_edges = set(pred_edges) & set(expected_edges)
    num_of_correctly_predicted_edges = []
    for i in range(0, len(pred_edges), step_size):
        num_of_correctly_predicted_edges.append(len(set(pred_edges[:i]) & all_correct_pred_edges))

    fig, ax = plt.subplots()
    ax.plot(np.arange(0, len(pred_edges), step_size), num_of_correctly_predicted_edges)
    fig.savefig(res_dir / 'num_of_correctly_predicted_edges.png', dpi=1200, transparent=True)


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

    # Make prediction using top n scoring edges
    new_edges = set(test_G.edges()) - set(train_G.edges())
    use_top_n_edges = len(new_edges) * 2
    pred_edges = prediction_vectorized(train_G, test_G, common_neighbors_vectorized, use_top_n_edges=use_top_n_edges)

    # Evaluate the prediction
    evaluation(pred_edges, new_edges)
