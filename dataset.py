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

def check_set(A, B):
    A = set(A)
    B = set(B)
    print(f"len A {len(A)} | len B {len(B)} | intersection {len(A.intersection(B))}")
    print(len(A.difference(B)))
    print(list(A.difference(B))[:10])
    print(len(B.difference(A)))
    print(list(B.difference(A))[:10])

def load_dates():
    # read in nodes and dates from cit-HepPh-dates.txt into a dictionary
    date_dict = {} 
    with open('dataset/cit-HepPh-dates.txt') as f:
        for line in f:
            # ignore lines starting with #
            if line[0] != '#':
                node, date = line.split()
                date_dict[int(node)] = date 
    check_set(date_dict.keys(), G.nodes(data=False))

def load_dataset(with_date=False, small=False):
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
    
    if small:
        # only use the first 10 nodes, for fast debugging purposes 
        G = G.subgraph(list(G.nodes)[:10])
    
    if not with_date:
        # load graph without publication dates 
        return G, {}                  
    else:
        # load the publication dates
        with open('dataset/cit-HepPh-dates.txt') as f:
            for line in f:
                if line[0] != '#': # ignore lines starting with #
                    node, date = line.split()
                    # if node is in graph, add date attribute
                    if int(node) in G.nodes():
                        G.nodes[int(node)]['date'] = str(date)

        # get the start and end date
        date_list = [] 
        for n,d in G.nodes(data=True):
            if 'date' in d:
                date_list.append(d['date'])
        start_date = min(date_list)
        end_date = max(date_list)
        return G, {'start_date': start_date, 'end_date': end_date}

def filter_graph(G):
    # filter out nodes without date
    return G.subgraph([n for n, d in G.nodes(data=True) if 'date' in d])

def graph_subset(G, start_date, end_date):
    # create graph subset containing nodes with date before date 
    # alternate: consider G.remove_nodes_from
    return G.subgraph([n for n, d in G.nodes(data=True) if (d['date'] >= start_date and d['date'] < end_date)])

if __name__ == '__main__':
    # test loading dataset
    print("\nTest loading dataset without dates")
    G, metadata = load_dataset(with_date=False)
    assert(G.number_of_nodes() == 34546 and G.number_of_edges() == 421578)
    print("Num Nodes:", G.number_of_nodes(), "Num Edges:", G.number_of_edges())
    print("First few nodes:", list(G.nodes(data=True))[:5])
    print("Example node:", G.nodes[9907233])

    print("\nTest loading dataset with dates")
    G, metadata = load_dataset(with_date=True)
    print("Num Nodes:", G.number_of_nodes(), "Num Edges:", G.number_of_edges())
    print("First few nodes:", list(G.nodes(data=True))[:5])
    print("Example node:", G.nodes[9907233])
    print("Metadata:", metadata)

    # test filter dataset 
    print("\nTest filtering out nodes without date")
    G = filter_graph(G)
    print("Num Nodes:", G.number_of_nodes(), "Num Edges:", G.number_of_edges())

    train_G = graph_subset(G, start_date='1992-01-01', end_date='1997-01-01')
    print("Num Nodes:", train_G.number_of_nodes(), "Num Edges:", train_G.number_of_edges())

    test_G = graph_subset(G, start_date='1997-01-01', end_date='2002-12-31')
    print("Num Nodes:", test_G.number_of_nodes(), "Num Edges:", test_G.number_of_edges())
