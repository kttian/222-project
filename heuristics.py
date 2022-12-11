'''
This file implements the prediction heuristics from in this paper 
https://www.cs.cornell.edu/home/kleinber/link-pred.pdf
Including:
- graph distance: length of shortest path between x and y
- common neighbors: count
- jaccard's coefficient
- adamic/adar
- preferential attachment 
- katz
- hitting time
- rooted pagerank
- simrank 
meta approaches 
- low-rank approximation 
- unseen bigrams 
- clustering 
'''

import logging
import time
from dataset import load_dataset_bitcoinotc, split_graph, load_dataset_collaboration

import networkx as nx
import numpy as np
import scipy as sp
import os

SAVE_DIR = 'scores_saved'

def graph_distance(G, x, y):
    '''
    length of shortest path between x and y
    '''
    try:
        return nx.shortest_path_length(G, source=x, target=y)
    except:
        return -1

def common_neighbors(G, u, v):
    """ Count the number of common neighbors.

    :param G:
    :param u:
    :param v:
    :return:
    """

    if nx.is_directed(G):
        return len(set(G.successors(u)) & set(G.successors(v)))
    else:
        return len(G.common_neighbors(u, v))


def common_neighbors_vectorized(G, nodelist=None, set_diag_zero=False, to_save_ds="", rerun=True):
    """ Gets common neighbors for all pairs of nodes in G.

    :param G:
    to_save_ds: if not empty, save the scores for this dataset
    :return: A matrix A where A[i, j] is the number of common successors between node i and node j.
    """

    # if nx.is_directed(G):
    # if save file exists, load scores
    if not rerun and os.path.exists(os.path.join(SAVE_DIR, f"{to_save_ds}_common_neighbors.npy")):
        logging.info(f"Loading scores from {SAVE_DIR}/{to_save_ds}_common_neighbors.npy")
        return np.load(os.path.join(SAVE_DIR, f"{to_save_ds}_common_neighbors.npy"))

    if nodelist is None:
        nodelist = sorted(G.nodes())
    adj_mat = nx.to_scipy_sparse_array(G, nodelist=nodelist)

    logging.info(f"Computing common neighbors")
    time_tic = time.perf_counter()

    scores = adj_mat.dot(adj_mat.T)
    if set_diag_zero:
        scores.setdiag(0)

    time_toc = time.perf_counter()
    logging.info(f"Finished computing common neighbors in {time_toc - time_tic} seconds")

    if to_save_ds:
        logging.info(f"Saving scores to {SAVE_DIR}/{to_save_ds}_common_neighbors.npy")
        np.save(f"{SAVE_DIR}/{to_save_ds}_common_neighbors.npy", scores.toarray())

    return scores.toarray()


def pagerank_vectorized(G, nodelist=None, to_save_ds="", rerun=True):
    '''
    Return the transition matrix used in pagerank
    '''
    if not rerun and os.path.exists(os.path.join(SAVE_DIR, f"{to_save_ds}_pagerank.npy")):
        logging.info(f"Loading scores from {SAVE_DIR}/{to_save_ds}_pagerank.npy")
        return np.load(os.path.join(SAVE_DIR, f"{to_save_ds}_pagerank.npy"))
    if nodelist is None:
        nodelist = sorted(G.nodes())
    scores = np.array(nx.google_matrix(G, alpha=0.9, nodelist=nodelist))
    if to_save_ds:
        logging.info(f"Saving scores to {SAVE_DIR}/{to_save_ds}_pagerank.npy")
        np.save(f"{SAVE_DIR}/{to_save_ds}_pagerank.npy", scores)
    return scores 


def jaccard_coefficient(G):
    '''
    jaccard's coefficient
    '''
    return nx.jaccard_coefficient(G)


def jaccard_coefficient_vectorized(G, nodelist=None, to_save_ds=""):
    """ Computes Jaccard's coefficient as defined in:

    Liben-Nowell, David, and Jon Kleinberg. “The Link Prediction Problem for Social Networks.” In Proceedings of the
    Twelfth International Conference on Information and Knowledge Management, 556–59. CIKM ’03.
    New York, NY, USA: Association for Computing Machinery, 2003. https://doi.org/10.1145/956863.956972.

    :param G:
    :param nodelist:
    :return: scores, a matrix where scores[i, j] is the Jaccard's coefficient between node i and node j
    """
    if os.path.exists(os.path.join(SAVE_DIR, f"{to_save_ds}_jaccard.npy")):
        logging.info(f"Loading scores from {SAVE_DIR}/{to_save_ds}_jaccard.npy")
        return np.load(os.path.join(SAVE_DIR, f"{to_save_ds}_jaccard.npy"))

    if nodelist is None:
        nodelist = sorted(G.nodes())

    logging.info(f"Computing Jaccard's coefficient")
    time_tic = time.perf_counter()

    # Remove all weights from the adjacency matrix because we do not need them
    adj_mat = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=None).astype(bool)

    num_nodes = len(nodelist)

    scores = np.zeros((len(nodelist), len(nodelist)))
    # Each iteration deals with all nodes u and u + i
    for i in range(1, num_nodes):
        # Each row vector contains all the neighbors of node u
        adj_mat_roll = sp.sparse.vstack((adj_mat[i:, :], adj_mat[:i, :]))
        # Use multiply to simulate an element-wise AND operation
        and_vector = adj_mat.multiply(adj_mat_roll).sum(axis=1)
        # Use addition to simulate an element-wise OR operation
        or_vector = (adj_mat + adj_mat_roll).astype(bool).sum(axis=1)
        score = np.divide(and_vector, or_vector, out=np.zeros_like(and_vector, dtype=float), where=or_vector != 0)
        for j in range(len(score)):
            scores[j, (j + i) % num_nodes] = score[j]

    time_toc = time.perf_counter()
    logging.info(f"Finished computing Jaccard's coefficient in {time_toc - time_tic:.2f} seconds")

    if to_save_ds:
        logging.info(f"Saving scores to {SAVE_DIR}/{to_save_ds}_jaccard.npy")
        np.save(f"{SAVE_DIR}/{to_save_ds}_jaccard.npy", scores)
    return scores


def adamic_adar(G):
    '''
    adamic/adar
    '''
    return nx.adamic_adar_index(G)


def adamic_adar_vectorized(G, nodelist=None, to_save_ds=""):
    """ Computes Adamic/Adar index as defined in:

        Liben-Nowell, David, and Jon Kleinberg. “The Link Prediction Problem for Social Networks.” In Proceedings of the
        Twelfth International Conference on Information and Knowledge Management, 556–59. CIKM ’03.
        New York, NY, USA: Association for Computing Machinery, 2003. https://doi.org/10.1145/956863.956972.

        :param G:
        :param nodelist:
        :return:
        """
    if os.path.exists(os.path.join(SAVE_DIR, f"{to_save_ds}_adamic_adar.npy")):
        logging.info(f"Loading scores from {SAVE_DIR}/{to_save_ds}_adamic_adar.npy")
        return np.load(os.path.join(SAVE_DIR, f"{to_save_ds}_adamic_adar.npy"))

    if nodelist is None:
        nodelist = sorted(G.nodes())

    logging.info(f"Computing Adamic/Adar coefficient")
    time_tic = time.perf_counter()

    # Remove all weights from the adjacency matrix because we do not need them
    adj_mat = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=None).astype(bool)

    nodeid_to_idx = {nodeid: idx for idx, nodeid in enumerate(nodelist)}

    # Precompute 1 / log(degree) for each node
    frequencies = np.zeros(len(nodelist))
    for nodeid in nodelist:
        out_deg = len(list(G.neighbors(nodeid)))
        if out_deg != 0 and out_deg != 1:
            frequencies[nodeid_to_idx[nodeid]] = 1 / np.log(out_deg)

    num_nodes = len(nodelist)

    scores = np.zeros((len(nodelist), len(nodelist)))
    # Each iteration deals with all nodes u and u + i
    for i in range(1, num_nodes):
        # Each row vector contains all the neighbors of node u
        adj_mat_roll = sp.sparse.vstack((adj_mat[i:, :], adj_mat[:i, :]))
        # Use multiply to simulate an element-wise AND operation
        and_matrix = adj_mat.multiply(adj_mat_roll)
        score = and_matrix.dot(frequencies)

        for j in range(len(score)):
            scores[j, (j + i) % num_nodes] = score[j]

    time_toc = time.perf_counter()
    logging.info(f"Finished computing Adamic/Adar coefficient in {time_toc - time_tic:.2f} seconds")

    if to_save_ds:
        logging.info(f"Saving scores to {SAVE_DIR}/{to_save_ds}_adamic_adar.npy")
        np.save(f"{SAVE_DIR}/{to_save_ds}_adamic_adar.npy", scores)
    return scores


def preferential_attachment_vectorized(G, nodelist=None):
    if nodelist is None:
        nodelist = sorted(G.nodes())

    logging.info(f"Computing preferential attachment")
    time_tic = time.perf_counter()

    # Remove all weights from the adjacency matrix because we do not need them
    adj_mat = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=None).astype(bool)

    num_neighbors = adj_mat.sum(axis=1)
    scores = np.matmul(num_neighbors[np.newaxis].T, num_neighbors[np.newaxis])

    time_toc = time.perf_counter()
    logging.info(f"Finished computing preferential attachment in {time_toc - time_tic:.2f} seconds")
    return scores

def katz(G):
    '''
    katz
    TODO: double check this one 
    '''
    return nx.katz_centrality(G)


def katz_vectorized(G, beta=0.05, nodelist=None, to_save_ds="", rerun = True):
    """ Computes Katz measure as defined in:

    Liben-Nowell, David, and Jon Kleinberg. “The Link Prediction Problem for Social Networks.” In Proceedings of the
    Twelfth International Conference on Information and Knowledge Management, 556–59. CIKM ’03.
    New York, NY, USA: Association for Computing Machinery, 2003. https://doi.org/10.1145/956863.956972.

    :param G:
    :param beta:
    :param nodelist:
    :return:
    """
    if not rerun and os.path.exists(os.path.join(SAVE_DIR, f"{to_save_ds}_katz.npy")):
        logging.info(f"Loading scores from {SAVE_DIR}/{to_save_ds}_katz.npy")
        return np.load(os.path.join(SAVE_DIR, f"{to_save_ds}_katz.npy"))

    # if nx.is_directed(G):
    if nodelist is None:
        nodelist = sorted(G.nodes())
    adj_mat = nx.to_scipy_sparse_array(G, nodelist=nodelist)

    logging.info(f"Performing large matrix multiplication")
    time_tic = time.perf_counter()
    identity = sp.sparse.identity(len(nodelist), format='csr')
    scores = sp.sparse.linalg.inv(identity - beta * adj_mat) - identity
    time_toc = time.perf_counter()
    logging.info(f"Finished large matrix multiplication in {time_toc - time_tic:.2f} seconds")

    if to_save_ds:
        logging.info(f"Saving scores to {SAVE_DIR}/{to_save_ds}_katz{beta}.npy")
        np.save(f"{SAVE_DIR}/{to_save_ds}_katz{beta}.npy", scores.toarray())
    return scores.toarray()
    # else:
    #     raise NotImplementedError("Not implemented for undirected graphs.")


def katz_0_1_vectorized(G, nodelist=None, to_save_ds="",rerun=True):
    return katz_vectorized(G, beta=0.1, nodelist=nodelist, to_save_ds=to_save_ds, rerun=rerun)

def katz_0_05_vectorized(G, nodelist=None, to_save_ds="",rerun=True):
    return katz_vectorized(G, beta=0.05, nodelist=nodelist, to_save_ds=to_save_ds, rerun=rerun)

def katz_0_01_vectorized(G, nodelist=None, to_save_ds="",rerun=True):
    return katz_vectorized(G, beta=0.01, nodelist=nodelist, to_save_ds=to_save_ds, rerun=rerun)

def katz_0_005_vectorized(G, nodelist=None,to_save_ds="", rerun=True):
    return katz_vectorized(G, beta=0.005, nodelist=nodelist,to_save_ds=to_save_ds,rerun=rerun)

def katz_0_0005_vectorized(G, nodelist=None, to_save_ds="", rerun=True):
    return katz_vectorized(G, beta=0.0005, nodelist=nodelist, to_save_ds=to_save_ds, rerun=rerun)


def hitting_time(G):
    '''
    hitting time
    '''
    pass 


if __name__ == '__main__':
    G, timelist = load_dataset_bitcoinotc()

    G_train, G_test = split_graph(G, 0.5, timelist)
    scores = common_neighbors_vectorized(G_train, to_save_ds="bitcoinotc_split0.5")
    scores = katz_0_05_vectorized(G_train, to_save_ds="bitcoinotc_split0.5")

    scores = jaccard_coefficient_vectorized(G_train, to_save_ds="bitcoinotc_split0.5")
    scores = adamic_adar_vectorized(G_train, to_save_ds="bitcoinotc_split0.5")

    G_train, G_test = split_graph(G, 0.8, timelist)
    scores = jaccard_coefficient_vectorized(G_train, to_save_ds="bitcoinotc_split0.8")
    scores = adamic_adar_vectorized(G_train, to_save_ds="bitcoinotc_split0.8")

    G, timelist = load_dataset_collaboration()

