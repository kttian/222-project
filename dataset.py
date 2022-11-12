'''
This file loads our datasets 
'''

import networkx as nx

# load the cit-HepPh dataset: a high-energy physics citation network
# https://snap.stanford.edu/data/cit-HepPh.html
# Nodes: 34546 Edges: 421578
# cit-HepPh.txt.gz:
#   is a list of edges - each row contains from_node and to_node
# cit-HepPh-dates.txt.gz:
#   each row contains node and date the paper was published 

def load_dataset():
    '''
    load in the cit-hep-ph dataset and return a networkx graph object
    '''
    # load the edges into a directed graph (DiGraph)
    G = nx.read_edgelist('data/cit-HepPh.txt.gz', create_using=nx.DiGraph())
    # load the publication dates
    with open('data/cit-HepPh-dates.txt.gz') as f:
        for line in f:
            node, date = line.split()
            G.nodes[node]['date'] = int(date)
    return G