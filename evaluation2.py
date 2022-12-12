from dataset import *
from heuristics import *
from evaluation import *
import time 

namelist = ["astro-ph", "cond-mat", "hep-th", "hep-ph", "gr-qc", "bitcoin-otc", "bitcoin-alpha", "amazon"]
# namelist = ["bitcoin-alpha"]
namelist = ["amazon"]
metric_list = [""]
split = 0.5

start_time = time.perf_counter()
for name in namelist:
    if name == "amazon":
        train_G, test_G = load_dataset_amazon_split(reduction=10)
    else:
        if name == "bitcoin-otc" or name == "bitcoin-alpha":
            G, date_list = load_dataset_bitcoinotc(name=name.split("-")[1])
        else:
            G, date_list = load_dataset_collaboration(name=name,OFFSET=0)
        train_G, test_G = split_graph(G, split, date_list)
    sorted_nodes = sorted(train_G.nodes())

    # for testing 
    new_edges = set(test_G.subgraph(train_G.nodes()).edges()) - set(train_G.edges())
    use_top_n_edges = min(50_000, len(new_edges))

    score_list=[]

    save_name = name + f"_offset_split{split}"
    save_name = ""
    rerun = True

    print(f"loading data done, {time.perf_counter() - start_time:.2f}s")
    # s_edges = len(new_edges)
    # s_nodes = len(train_G.nodes()) ** 2 
    # print(f"{name}: test edges {s_edges}, nodes^2 {s_nodes}, ratio {s_edges/(s_nodes-s_edges):.3f}%")

    # exploring Katz
    # score_list.append(("katz_0_1", katz_0_1_vectorized(train_G, sorted_nodes, to_save_ds=name, rerun=rerun)))
    # score_list.append(("katz_0_05", katz_0_05_vectorized(train_G, sorted_nodes, to_save_ds=name, rerun=rerun)))
    # score_list.append(("katz_0_01", katz_0_01_vectorized(train_G, sorted_nodes, to_save_ds=name, rerun=rerun)))
    # score_list.append(("katz_0_005", katz_0_005_vectorized(train_G, sorted_nodes, to_save_ds=name, rerun=rerun)))
    # score_list.append(("katz_0_0005", katz_0_0005_vectorized(train_G, sorted_nodes, to_save_ds=name, rerun=rerun)))

    score_list.append(("cn", common_neighbors_vectorized(train_G, sorted_nodes, to_save_ds=save_name, rerun=rerun)))
    print(f"cn done, {time.perf_counter() - start_time:.2f}s")
    score_list.append(("katz_0_05", katz_0_05_vectorized(train_G, sorted_nodes, to_save_ds=save_name, rerun=rerun)))
    print(f"katz done, {time.perf_counter() - start_time:.2f}s")
    score_list.append(("katz_0_005", katz_0_005_vectorized(train_G, sorted_nodes, to_save_ds=save_name, rerun=rerun)))
    print(f"katz done, {time.perf_counter() - start_time:.2f}s")
    score_list.append(("katz_0_0005", katz_0_0005_vectorized(train_G, sorted_nodes, to_save_ds=save_name, rerun=rerun)))
    print(f"katz done, {time.perf_counter() - start_time:.2f}s")
    score_list.append(("jaccard", jaccard_coefficient_vectorized(train_G, sorted_nodes, to_save_ds=save_name, rerun=rerun)))
    print(f"jaccard done, {time.perf_counter() - start_time:.2f}s")
    score_list.append(("adamic", adamic_adar_vectorized(train_G, sorted_nodes, to_save_ds=save_name, rerun=rerun)))
    print(f"adamic done, {time.perf_counter() - start_time:.2f}s")
    score_list.append(("pref_a", preferential_attachment_vectorized(train_G, sorted_nodes)))
    print(f"pref_a done, {time.perf_counter() - start_time:.2f}s")

    # score_list.append(("pagerank", pagerank_vectorized(train_G, sorted_nodes,to_save_ds=save_name, rerun=rerun)))

    # train_scores_cn.shape, train_scores_katz.shape

    # looping across different heuristics 
    for scores in score_list:
        pred_edges = prediction_vectorized(scores[1], train_G, use_top_n_edges=use_top_n_edges)
        f1 = compute_f1(pred_edges, new_edges)

        print(f"data {name}, score {scores[0]}, f1 {f1:.8f}")

print(f"all done, {time.perf_counter() - start_time:.2f}s")