'''
This file loads our datasets 
TOOD: add links for downloading into dataset folder
'''

import datetime
import networkx as nx
import numpy as np
import pandas as pd

def check_set(A, B):
    A = set(A)
    B = set(B)
    print(f"len A {len(A)} | len B {len(B)} | intersection {len(A.intersection(B))}")
    print(len(A.difference(B)))
    print(list(A.difference(B))[:10])
    print(len(B.difference(A)))
    print(list(B.difference(A))[:10])


# functions for loading and processing the cit-HepPh dataset: 
# a high-energy physics citation network
# https://snap.stanford.edu/data/cit-HepPh.html
# Nodes: 34546 Edges: 421578
# cit-HepPh.txt.gz:
#   is a list of edges - each row contains from_node and to_node
# cit-HepPh-dates.txt.gz:
#   each row contains node and date the paper was published 

def load_dates_cit_hep_ph():
    # read in nodes and dates from cit-HepPh-dates.txt into a dictionary
    date_dict = {} 
    with open('dataset/cit-HepPh-dates.txt') as f:
        for line in f:
            # ignore lines starting with #
            if line[0] != '#':
                node, date = line.split()
                date_dict[int(node)] = date 
    check_set(date_dict.keys(), G.nodes(data=False))

def load_dataset_cit_hep_ph(small=-1):
    '''
    load in the cit-hep-ph dataset
    
    returns
    - a networkx graph object
    - graph metadata, including the start and end date  
    '''
    # load the edges into a directed graph (DiGraph)
    # TODO: add support for .gz files 
    G = nx.read_edgelist('dataset/cit-HepPh.txt', create_using=nx.DiGraph(),
                         comments='#', nodetype=int, data=True)
    
    if small > 0:
        # only use the first 'small' number of nodes, for fast debugging purposes 
        G = G.subgraph(list(G.nodes)[:small])
    
    # load the publication dates
    # and add date as a node attribute 
    node2date = {}
    with open('dataset/cit-HepPh-dates.txt') as f:
        for line in f:
            if line[0] != '#': # ignore lines starting with #
                node, date = line.split()
                date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
                node2date[int(node)] = int(date_obj.timestamp())

    # add date as edge attribute 
    for u, v, e in G.edges(data=True):
        if u in node2date:
            e['time'] = node2date[u]

    if small > 0:
        date_list = [a['time'] for s,t,a in G.edges(data=True)]
    else:
        date_list = list(node2date.values())
    
    # return graph, time list 
    return G, date_list 

def filter_prolific_authors(G, kappa=3):
    """ Filter out authors who have at least written a minimum number of papers.

    Liben-Nowell, David, and Jon Kleinberg. “The Link Prediction Problem for Social Networks.” In Proceedings of the
    Twelfth International Conference on Information and Knowledge Management, 556–59. CIKM ’03.
    New York, NY, USA: Association for Computing Machinery, 2003. https://doi.org/10.1145/956863.956972.
.
    :param G:
    :param kappa:
    :return:
    """
    return G.subgraph([n for n in G.nodes() if len(list(G.neighbors(n))) >= kappa])

def test_loading(G, timelist):
    # test loading dataset
    print("...loading dataset")
    # G, timelist = load_dataset(small=small)
    print("Num Nodes:", G.number_of_nodes(), "Num Edges:", G.number_of_edges())
    print("Time Qtles:", np.quantile(timelist, [0.25, 0.5, 0.75]))
    node_list = list(G.nodes(data=True))
    edge_list = list(G.edges(data=True))
    print("First few nodes:", node_list[:5])
    print("First few edges:", edge_list[:5])

    print("...splitting dataset")
    train_G, test_G = split_graph(G, timelist, 0.5)
    print("Num Nodes:", train_G.number_of_nodes(), "Num Edges:", train_G.number_of_edges())
    print("Num Nodes:", test_G.number_of_nodes(), "Num Edges:", test_G.number_of_edges())


# bitcoin data
# https://snap.stanford.edu/data/soc-sign-bitcoinotc.html
# Nodes	5,881 Edges	35,592
# Range of edge weight	-10 to +10
# Percentage of positive edges	89%
def load_dataset_bitcoinotc(small=-1):
    '''
    Returns:
        G: full graph, which is also the test graph
        train_G: train graphs
    '''
    df = pd.read_csv("dataset/soc-sign-bitcoinotc.csv.gz", compression='gzip', header=None)
    df.columns = ['source', 'target', 'rating', 'time']
    G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=['rating', 'time'], create_using=nx.DiGraph())
    if small > 0:
        G = G.subgraph(list(G.nodes)[:small])

    if small > 0:
        date_list = [a['time'] for s,t,a in G.edges(data=True)]
    else:
        date_list = list(df['time'])

    return G, date_list

def split_graph(G, time_list, split_quantile):
    '''
    Takes in a graph, range of times, and a split quantile (e.g. 0.5)
    Returns:
        train graph, test graph
    '''
    # for some reason we need to apply a filter on the citations network, since 
    # not all papers have a date 
    G = filter_graph(G)
    time_split = np.quantile(time_list, [split_quantile])[0]
    print("TIME SPLIT", time_split)
    train_G = graph_subset(G, start_date=np.min(time_list), end_date=time_split)
    return train_G, G 

def filter_graph(G):
    """ Filter out edges without dates.

    :param G:
    :return:
    """
    # alternate: we could consider G.remove_nodes_from instead of subgraph
    edge_list = [(s,t) for s,t,a in G.edges(data=True) if 'time' in a]
    return G.edge_subgraph(edge_list)

def graph_subset(G, start_date, end_date):
    # create graph subset containing nodes with time within given start and end 
    # (s,t,a) is a tuple of source, target, and attribute
    edge_list = [(s,t) for s,t,a in G.edges(data=True) if start_date <= a['time'] < end_date]
    return G.edge_subgraph(edge_list)


if __name__ == '__main__':
    print("\nbitcoin graph")
    G, timelist = load_dataset_bitcoinotc(small=500)
    test_loading(G, timelist)

    print("\ncitation graph")
    G, timelist = load_dataset_cit_hep_ph(small=100)
    test_loading(G, timelist)
