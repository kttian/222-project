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
from dataset import load_dataset_bitcoinotc

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

    if nx.is_directed(G):
        # if save file exists, load scores
        if not rerun and os.path.exists(os.path.join(SAVE_DIR, f"{to_save_ds}_common_neighbors.npy")):
            logging.info(f"Loading scores from {SAVE_DIR}/{to_save_ds}_common_neighbors.npy")
            return np.load(os.path.join(SAVE_DIR, f"{to_save_ds}_common_neighbors.npy"))

        if nodelist is None:
            nodelist = sorted(G.nodes())
        adj_mat = nx.adjacency_matrix(G, nodelist=nodelist)

    logging.info(f"Performing large matrix multiplication")
    time_tic = time.perf_counter()
    scores = adj_mat.dot(adj_mat.T)
    if set_diag_zero:
        scores.setdiag(0)
    time_toc = time.perf_counter()
    logging.info(f"Finished large matrix multiplication in {time_toc - time_tic:.2f} seconds")

    if to_save_ds:
        logging.info(f"Saving scores to {SAVE_DIR}/{to_save_ds}_common_neighbors.npy")
        np.save(f"{SAVE_DIR}/{to_save_ds}_common_neighbors.npy", scores.toarray())
    return scores.toarray()


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
    nodeid_to_idx = {nodeid: idx for idx, nodeid in enumerate(nodelist)}
    if nx.is_directed(G):
        logging.info(f"Input graph is directed; converting to undirected graph")
        G = G.to_undirected()
    logging.info(f"Computing Jaccard's coefficient")
    time_tic = time.perf_counter()
    scores = np.zeros((len(nodelist), len(nodelist)))
    for u, v, p in nx.jaccard_coefficient(G):
        scores[nodeid_to_idx[u], nodeid_to_idx[v]] = p
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
    nodeid_to_idx = {nodeid: idx for idx, nodeid in enumerate(nodelist)}
    if nx.is_directed(G):
        logging.info(f"Input graph is directed; converting to undirected graph")
        G = G.to_undirected()
    logging.info(f"Computing Adamic/Adar coefficient")
    time_tic = time.perf_counter()
    scores = np.zeros((len(nodelist), len(nodelist)))
    for u, v, p in nx.jaccard_coefficient(G):
        scores[nodeid_to_idx[u], nodeid_to_idx[v]] = p
    time_toc = time.perf_counter()
    logging.info(f"Finished computing Adamic/Adar coefficient in {time_toc - time_tic:.2f} seconds")

    if to_save_ds:
        logging.info(f"Saving scores to {SAVE_DIR}/{to_save_ds}_adamic_adar.npy")
        np.save(f"{SAVE_DIR}/{to_save_ds}_adamic_adar.npy", scores)
    return scores


def preferential_attachment(G):
    '''
    preferential attachment 
    '''
    return nx.preferential_attachment(G)

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

    if nx.is_directed(G):
        if nodelist is None:
            nodelist = sorted(G.nodes())
        adj_mat = nx.adjacency_matrix(G, nodelist=nodelist)

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
    else:
        raise NotImplementedError("Not implemented for undirected graphs.")


def katz_0_05_vectorized(G, nodelist=None, to_save_ds=""):
    return katz_vectorized(G, beta=0.05, nodelist=nodelist, to_save_ds=to_save_ds)


def katz_0_005_vectorized(G, nodelist=None):
    return katz_vectorized(G, beta=0.005, nodelist=nodelist)


def katz_0_0005_vectorized(G, nodelist=None):
    return katz_vectorized(G, beta=0.0005, nodelist=nodelist)


def hitting_time(G):
    '''
    hitting time
    '''
    pass 


if __name__ == '__main__':
    G, timelist = load_dataset_bitcoinotc()
    scores = common_neighbors_vectorized(G, to_save_ds="bitcoinotc")
    scores = katz_0_05_vectorized(G, to_save_ds="bitcoinotc")
    # scores = jaccard_coefficient_vectorized(G, to_save_ds="bitcoinotc")
    # scores = adamic_adar_vectorized(G, to_save_ds="bitcoinotc")
