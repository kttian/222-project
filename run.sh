#!/bin/zsh

for dataset in collaboration_astro-ph collaboration_cond-mat collaboration_gr-qc collaboration_hep-ph collaboration_hep-th bitcoinotc; do
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --test_no_add_nodes --heuristic cn                      &
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --test_no_add_nodes --heuristic jaccard                 &
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --test_no_add_nodes --heuristic adamic_adar             &
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --test_no_add_nodes --heuristic preferential_attachment &
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --test_no_add_nodes --heuristic katz-0_05               &
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --test_no_add_nodes --heuristic katz-0_005              &
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --test_no_add_nodes --heuristic katz-0_0005             &
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --test_no_add_nodes --heuristic pagerank                &
done