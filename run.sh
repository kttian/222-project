#!/bin/zsh

for dataset in collaboration_astro-ph collaboration_cond-mat collaboration_gr-qc collaboration_hep-ph collaboration_hep-th bitcoinotc; do
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --heuristic cn                      --test_no_add_nodes &
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --heuristic jaccard                 --test_no_add_nodes &
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --heuristic adamic_adar             --test_no_add_nodes &
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --heuristic preferential_attachment --test_no_add_nodes &
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --heuristic katz-0_05               --test_no_add_nodes &
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --heuristic katz-0_005              --test_no_add_nodes &
  python evaluation_exp.py --dataset $dataset --dataset_split_quantile 0.5 --heuristic katz-0_0005             --test_no_add_nodes &
done