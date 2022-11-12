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

import networkx as nx

def graph_distance(G):
    '''
    length of shortest path between x and y
    '''
    return nx.shortest_path_length(G)

def common_neighbors(G):
    '''
    count number of common neighbors
    '''
    return nx.common_neighbors(G)

def jaccard_coefficient(G):
    '''
    jaccard's coefficient
    '''
    return nx.jaccard_coefficient(G)

def adamic_adar(G):
    '''
    adamic/adar
    '''
    return nx.adamic_adar_index(G)

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

def hitting_time(G):
    '''
    hitting time
    '''
    pass 

