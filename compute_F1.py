from dataset import *
from heuristics import *
from evaluation import *

# load bitcoin dataset 
'''
G, date_list = load_dataset_bitcoinotc()
train_G, test_G = split_graph(G, 0.5, date_list)

sorted_nodes = sorted(train_G.nodes())
train_scores_cn = common_neighbors_vectorized(train_G, nodelist=sorted_nodes)
train_scores_katz = katz_vectorized(train_G, nodelist=sorted_nodes)
# train_scores_cn.shape, train_scores_katz.shape

# looping across different heuristics 
for scores in [train_scores_cn, train_scores_katz]:
    new_edges = set(test_G.subgraph(train_G.nodes()).edges()) - set(train_G.edges())
    use_top_n_edges = min(50_000, len(new_edges))

    pred_edges = prediction_vectorized(scores, train_G, use_top_n_edges=use_top_n_edges)
    f1 = compute_f1(pred_edges, new_edges)
    # print f1 
    print(f1)
'''

# load collaboration network
namelist = ["astro-ph", "cond-mat", "hep-th", "hep-ph", "gr-qc"]
# TODO loop through namelist 
G, date_list = load_dataset_collaboration(name=namelist[0])
train_G, test_G = split_graph(G, 0.5, date_list)

sorted_nodes = sorted(train_G.nodes())
train_scores_cn = common_neighbors_vectorized(train_G, nodelist=sorted_nodes)
train_scores_katz = katz_vectorized(train_G, nodelist=sorted_nodes)
# train_scores_cn.shape, train_scores_katz.shape

# looping across different heuristics 
for scores in [train_scores_cn, train_scores_katz]:
    new_edges = set(test_G.subgraph(train_G.nodes()).edges()) - set(train_G.edges())
    use_top_n_edges = min(50_000, len(new_edges))

    pred_edges = prediction_vectorized(scores, train_G, use_top_n_edges=use_top_n_edges)
    f1 = compute_f1(pred_edges, new_edges)
    # print f1 
    print(f1)