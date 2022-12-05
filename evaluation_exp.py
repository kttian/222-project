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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

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
    np.nan_to_num(edge_scores, copy=False, nan=max_edge_score + 1)
    plt.hist(edge_scores)
    plt.hist(no_edge_scores)
    plt.show()

    '''
    alternate? for every edge in test graph that wasn't already in the train graph,
    see whether scores are predictive 
    '''


def score_vectorized(G_train, heuristic_fn_vec, nodelist=None):
    '''
    Compute the score for every pair of nodes in the train graph
    To be used in scoring test edges 
    Returns:
        scores: NxN matrix of edge scores 
    '''
    # Obtain scores for all edges
    if nodelist is None:
        nodelist = np.array(sorted(G_train.nodes()))
    scores = heuristic_fn_vec(G_train, nodelist=nodelist)
    return scores


def prediction_vectorized(scores, train_G, use_top_n_edges=None):
    """ Predicts edges in G_test using the heuristic function.
    Converts the scores computed with a heuristic function into edge predictions
    via the method from the Kleinberg paper, which takes the top n
    scoring edges to predict the n new edges in the test graph

    :param scores: NxN matrix of edge scores
    :param train_G: train graph
    :param use_top_n_edges: Integer, top n scoring edges to use for prediction. If None, use number
        of new edges in G_test.
    :return:
    """
    assert (use_top_n_edges is not None)

    sorted_nodes = np.array(sorted(train_G.nodes()))

    # Sort edges scores
    logging.info(f"Sorting scores")
    time_tic = time.perf_counter()
    # argsort the scores in ascending order to get the u and v endpoints of sorted edges 
    sorted_scoring_edges_u, sorted_scoring_edges_v = np.unravel_index(np.argsort(scores, axis=None), shape=scores.shape)
    # reverse to get descending order
    sorted_scoring_edges = zip(
        reversed(sorted_nodes[sorted_scoring_edges_u]),
        reversed(sorted_nodes[sorted_scoring_edges_v])
    )
    time_toc = time.perf_counter()
    logging.info(f"Sorted scores in {time_toc - time_tic:.2f} seconds")

    # Obtain new edges in test graph
    train_G_edges_set = set(train_G.edges())

    # Need to remove the edges that already exist in the train graph
    logging.info(f"Removing existing edges")
    time_tic = time.perf_counter()
    pred_edges = []
    for u, v in sorted_scoring_edges:
        if len(pred_edges) >= use_top_n_edges:
            break
        if (u, v) not in train_G_edges_set:
            pred_edges.append((u, v))
    time_toc = time.perf_counter()
    logging.info(f"Removed existing edges in {time_toc - time_tic:.2f} seconds")

    return pred_edges


def evaluation(pred_edges, expected_edges, step_size=1000):
    """ Evaluate the accuracy of the prediction.

    :param pred_edges:
    :param expected_edges:
    :param step_size:
    :return:
    """
    all_correct_pred_edges = set(pred_edges) & set(expected_edges)
    num_of_correctly_predicted_edges = []
    for i in range(0, len(pred_edges), step_size):
        num_of_correctly_predicted_edges.append(len(set(pred_edges[:i]) & all_correct_pred_edges))
    predicted_edges_acc = np.array(num_of_correctly_predicted_edges) / len(expected_edges)

    return predicted_edges_acc


def plot_evaluation(
        acc_temporal,
        acc_random,
        step_size=1000,
        project_dir=Path.cwd(),
        dataset_name="",
        heuristic_name="",
        dataset_split_quantile=0.0,
        ylim=None
):
    """ Plot the evaluation result.
    Plot the accuracy of the prediction vs top n edges

    :param predicted_edges_acc:
    :param step_size:
    :param project_dir:
    :param dataset_name:
    :param heuristic_name:
    :return:
    """
    assert(isinstance(acc_random, list))

    # Create result directory
    res_dir = project_dir / 'res' / dataset_name / f'split-{dataset_split_quantile}'
    if not res_dir.exists():
        res_dir.mkdir(parents=True)

    # Plot the performance
    fig, ax = plt.subplots()
    x = np.arange(len(acc_temporal)) * step_size
    sns.lineplot(x=x, y=acc_temporal, label="Temporal", ax=ax)
    sns.lineplot(x=np.tile(x, len(acc_random)), y=np.array(acc_random).flatten(), label="Random", ax=ax)

    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(f"Accuracy - {heuristic_name} - Split - {dataset_split_quantile}")
    ax.set_ylabel('correctly predicted edges / number of new edges')
    ax.set_xlabel(f'Top n scoring edges, binned by {step_size}')
    fig.savefig(res_dir / f'prediction_acc-{heuristic_name}.png', dpi=300, transparent=True)


def plot_pos_neg_scores(
        scores,
        G_train,
        G_test,
        project_dir=Path.cwd(),
        dataset_name="",
        dataset_split_quantile=0.0,
        fig_name="",
):
    '''
    Obtain list of scores for new edges and non-edges in the test graph
    in order to compute metrics such as distance between distributions 
    '''
    # Create result directory
    res_dir = project_dir / 'res' / dataset_name / f'split-{dataset_split_quantile}'
    if not res_dir.exists():
        res_dir.mkdir(parents=True)

    sorted_nodes = np.array(sorted(G_train.nodes()))

    test_adj_mat = nx.to_numpy_array(G_test.subgraph(sorted_nodes), nodelist=sorted_nodes)
    test_inv_adj_mat = np.logical_not(test_adj_mat).astype(int)

    train_adj_mat = nx.to_numpy_array(G_train.subgraph(sorted_nodes), nodelist=sorted_nodes)
    train_inv_adj_mat = np.logical_not(train_adj_mat).astype(int)

    # new edges: in test but not in train
    new_edges = np.logical_and(test_adj_mat, train_inv_adj_mat).astype(int)
    # non edges: never appears in test (or train)
    non_edges = test_inv_adj_mat

    positive_scores = scores.flatten()[np.where(new_edges.flatten() == 1)[0]]
    negative_scores = scores.flatten()[np.where(non_edges.flatten() == 1)[0]]

    fig, ax = plt.subplots()
    max_score = max(np.max(positive_scores), np.max(negative_scores))
    min_score = min(np.min(positive_scores), np.min(negative_scores))
    bins = np.concatenate((np.linspace(min_score, max_score, 100), [max_score]))
    ax.hist(negative_scores, bins=bins, density=True, alpha=0.75, label='non-edges')
    ax.hist(positive_scores, bins=bins, density=True, alpha=0.75, label='new edges')
    ax.set_title(f"Score distribution - {fig_name} (binsize = {bins[1] - bins[0]:.2f})")
    # ax.set_ylim(0, 1)
    ax.set_ylabel('Density')
    ax.set_xlabel(f'Score')
    ax.legend()
    fig.savefig(res_dir / f'score_distribution-{fig_name}.png', dpi=300, transparent=True)

    logging.info(f"Positive score - N:    {np.size(positive_scores)}")
    logging.info(f"Positive score - Max:  {np.max(positive_scores)}")
    logging.info(f"Positive score - Min:  {np.min(positive_scores)}")
    logging.info(f"Positive score - Mean: {np.mean(positive_scores)}")
    logging.info(f"Positive score - Std:  {np.std(positive_scores)}\n")

    logging.info(f"Negative score - N:    {np.size(negative_scores)}")
    logging.info(f"Negative score - Max:  {np.max(negative_scores)}")
    logging.info(f"Negative score - Min:  {np.min(negative_scores)}")
    logging.info(f"Negative score - Mean: {np.mean(negative_scores)}")
    logging.info(f"Negative score - Std:  {np.std(negative_scores)}")

    # TODO: can compute any other distribution statistics on 
    # positive scores vs negative scores here


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Experiments to compare heuristic performance between random cut and temporal cut")
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset to load',
                        choices=[
                            'bitcoinotc',
                            'collaboration_astro-ph',
                            'collaboration_cond-mat',
                            'collaboration_gr-qc',
                            'collaboration_hep-ph',
                            'collaboration_hep-th',
                        ],
                        )
    parser.add_argument('--dataset_split_quantile', type=float, default=0.5,
                        help='quantile to split between train and test dataset')
    parser.add_argument('--heuristic', type=str, required=True,
                        help='heuristic to use',
                        choices=[
                            'cn',
                            'jaccard',
                            'adamic_adar',
                            'katz-0_05',
                            'katz-0_005',
                            'katz-0_0005',
                        ],
                        )
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='number of seeds to use for random cut',
                        )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(filename)s %(funcName)s(%(lineno)d) %(message)s',
    )

    # Load heuristic function
    if args.heuristic == 'cn':
        heuristic_fn_vec = common_neighbors_vectorized
    elif args.heuristic == 'jaccard':
        heuristic_fn_vec = jaccard_coefficient_vectorized
    elif args.heuristic == 'adamic_adar':
        heuristic_fn_vec = adamic_adar_vectorized
    elif args.heuristic == 'katz-0_05':
        heuristic_fn_vec = katz_0_05_vectorized
    elif args.heuristic == 'katz-0_005':
        heuristic_fn_vec = katz_0_005_vectorized
    elif args.heuristic == 'katz-0_0005':
        heuristic_fn_vec = katz_0_0005_vectorized
    else:
        raise ValueError(f"Unknown heuristic {args.heuristic}")

    # Load dataset
    if args.dataset == 'bitcoinotc':
        G, date_list = load_dataset_bitcoinotc()
        train_G, test_G = split_graph(G, args.dataset_split_quantile, time_list=date_list)
    elif args.dataset.startswith('collaboration'):
        name = args.dataset.split('_')[1]
        G = load_dataset_collaboration(name)
        G = filter_prolific_authors(G)
        train_G = graph_subset(G, 1994, 1997)
        test_G = graph_subset(G, 1994, 2000)
    else:
        raise ValueError(f"Unknown dataset {args.exp_name}")

    num_dropped_edges_for_temporal = test_G.number_of_edges() - train_G.number_of_edges()
    logging.info(f"Number of dropped edges for temporal cut: {num_dropped_edges_for_temporal}")

    # Limit the top n scores to consider for performance reasons
    top_n_edges = min(50_000, test_G.number_of_edges())

    # Obtain scores with train graph created by temporal cuts
    scores = score_vectorized(train_G, heuristic_fn_vec)
    pred_edges = prediction_vectorized(scores, train_G, use_top_n_edges=top_n_edges)
    # TODO: train_G and test_G might be a MultiGraph
    new_edges = set(test_G.edges()) - set(train_G.edges())
    acc_temporal = evaluation(pred_edges, new_edges)
    plot_pos_neg_scores(
        scores,
        train_G,
        test_G,
        dataset_name=args.dataset,
        dataset_split_quantile=args.dataset_split_quantile,
        fig_name=f'{args.heuristic}-temporal',
    )

    # Obtain scores with train graph created by random cuts
    acc_random = []
    for seed in range(args.num_seeds):
        rng = np.random.default_rng(seed)
        train_G = drop_random_edges(test_G, num_edges=num_dropped_edges_for_temporal, inplace=False, rng=rng)
        scores = score_vectorized(train_G, heuristic_fn_vec)
        pred_edges = prediction_vectorized(scores, train_G, use_top_n_edges=top_n_edges)
        # TODO: train_G and test_G might be a MultiGraph
        new_edges = set(test_G.edges()) - set(train_G.edges())
        acc_random.append(evaluation(pred_edges, new_edges))

    # Evaluate the prediction
    plot_evaluation(
        acc_temporal,
        acc_random,
        dataset_name=args.dataset,
        heuristic_name=args.heuristic,
        dataset_split_quantile=args.dataset_split_quantile,
    )

    # Plot pos neg scores of random cut
    plot_pos_neg_scores(
        scores,
        train_G,
        test_G,
        dataset_name=args.dataset,
        dataset_split_quantile=args.dataset_split_quantile,
        fig_name=f'{args.heuristic}-random',
    )
