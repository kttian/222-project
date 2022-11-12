'''
This file performs evaluation on our datasets and heuristics
Given a list of datasets and a list of heuristics, computes the performance table

Performance metrics:
- accuracy: # edges predicted correctly / # missing edges in test set
'''
import argparse
import networkx as nx 
import matplotlib.pyplot as plt 
from heuristics import * 
from dataset import * 

def evaluation(G_test, G_train, heuristic_fn):
    '''
    for every pair of vertices in train graph without an existing edge,
    predict whether a new edge will form there in the future
    problem: struggle with low # of actual future edges, also slow 
    '''
    edge_scores = []
    no_edge_scores = []
    for u in G_train.nodes():
        for v in G_train.nodes():
            if u != v:
                score = heuristic_fn(G_train, u, v)
                if G_test.has_edge(u,v):
                    edge_scores.append(score)
                else:
                    no_edge_scores.append(score)
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

def accuracy(G_test, G_pred):
    '''
    accuracy: # edges predicted correctly / # missing edges in test set
    '''
    # TODO: how to turn into a prediction?
    pass 

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, required=True)
    # parser.add_argument('--model', type=str, required=True)
    # parser.add_argument('--output', type=str, required=True)
    # parser.add_argument('--data', type=str, required=True)
    # parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--batch_size', type=int, default=32)
    # args = parser.parse_args()
    G, metadata = load_dataset(with_date=True, small=1000)
    G = filter_graph(G)
    train_G = graph_subset(G, start_date='1992-01-01', end_date='1997-01-01')
    test_G = graph_subset(G, start_date='1997-01-01', end_date='2002-12-31')
    
    evaluation(test_G, train_G, graph_distance)

