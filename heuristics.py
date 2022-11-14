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

import networkx as nx
import numpy as np
import scipy as sp

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


def common_neighbors_vectorized(G, nodelist=None):
    """ Gets common neighbors for all pairs of nodes in G.

    :param G:
    :return: A matrix A where A[i, j] is the number of common successors between node i and node j.
    """

    if nx.is_directed(G):
        if nodelist is None:
            nodelist = sorted(G.nodes())
        adj_mat = nx.adjacency_matrix(G, nodelist=nodelist)

        logging.info(f"Performing large matrix multiplication")
        time_tic = time.perf_counter()
        scores = adj_mat.dot(adj_mat.T)
        scores.setdiag(0)
        time_toc = time.perf_counter()
        logging.info(f"Finished large matrix multiplication in {time_toc - time_tic:.2f} seconds")

        return scores.toarray()
    else:
        raise NotImplementedError("Not implemented for undirected graphs.")


def jaccard_coefficient(G):
    '''
    jaccard's coefficient
    '''
    return nx.jaccard_coefficient(G)


def jaccard_coefficient_vectorized(G, nodelist=None):
    """ Computes Jaccard's coefficient as defined in:

    Liben-Nowell, David, and Jon Kleinberg. “The Link Prediction Problem for Social Networks.” In Proceedings of the
    Twelfth International Conference on Information and Knowledge Management, 556–59. CIKM ’03.
    New York, NY, USA: Association for Computing Machinery, 2003. https://doi.org/10.1145/956863.956972.

    :param G:
    :param nodelist:
    :return:
    """
    if nodelist is None:
        nodelist = sorted(G.nodes())
    logging.info(f"Computing Jaccard's coefficient")
    time_tic = time.perf_counter()
    scores = np.zeros(len(nodelist), len(nodelist))
    for u, v, p in nx.jaccard_coefficient(G):
        scores[u, v] = p
    time_toc = time.perf_counter()
    logging.info(f"Finished computing Jaccard's coefficient in {time_toc - time_tic:.2f} seconds")
    return scores


def adamic_adar(G):
    '''
    adamic/adar
    '''
    return nx.adamic_adar_index(G)


def adamic_adar_vectorized(G, nodelist=None):
    """ Computes Adamic/Adar index as defined in:

        Liben-Nowell, David, and Jon Kleinberg. “The Link Prediction Problem for Social Networks.” In Proceedings of the
        Twelfth International Conference on Information and Knowledge Management, 556–59. CIKM ’03.
        New York, NY, USA: Association for Computing Machinery, 2003. https://doi.org/10.1145/956863.956972.

        :param G:
        :param nodelist:
        :return:
        """
    if nodelist is None:
        nodelist = sorted(G.nodes())
    logging.info(f"Computing Adamic/Adar coefficient")
    time_tic = time.perf_counter()
    scores = np.zeros(len(nodelist), len(nodelist))
    for u, v, p in nx.adamic_adar_index(G):
        scores[u, v] = p
    time_toc = time.perf_counter()
    logging.info(f"Finished computing Adamic/Adar coefficient in {time_toc - time_tic:.2f} seconds")
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


def katz_vectorized(G, beta=0.05, nodelist=None):
    """ Computes Katz measure as defined in:

    Liben-Nowell, David, and Jon Kleinberg. “The Link Prediction Problem for Social Networks.” In Proceedings of the
    Twelfth International Conference on Information and Knowledge Management, 556–59. CIKM ’03.
    New York, NY, USA: Association for Computing Machinery, 2003. https://doi.org/10.1145/956863.956972.

    :param G:
    :param beta:
    :param nodelist:
    :return:
    """
    if nx.is_directed(G):
        if nodelist is None:
            nodelist = sorted(G.nodes())
        adj_mat = nx.adjacency_matrix(G, nodelist=nodelist)

        logging.info(f"Performing large matrix multiplication")
        time_tic = time.perf_counter()
        identity = sp.sparse.identity(len(nodelist), format='csr')
        scores = sp.sparse.linalg.inv(identity - beta * adj_mat) - identity
        scores.setdiag(0)
        time_toc = time.perf_counter()
        logging.info(f"Finished large matrix multiplication in {time_toc - time_tic:.2f} seconds")

        return scores.toarray()
    else:
        raise NotImplementedError("Not implemented for undirected graphs.")


def katz_0_05_vectorized(G, nodelist=None):
    return katz_vectorized(G, beta=0.05, nodelist=nodelist)


def katz_0_005_vectorized(G, nodelist=None):
    return katz_vectorized(G, beta=0.005, nodelist=nodelist)


def katz_0_0005_vectorized(G, nodelist=None):
    return katz_vectorized(G, beta=0.0005, nodelist=nodelist)


def hitting_time(G):
    '''
    hitting time
    '''
    pass 

