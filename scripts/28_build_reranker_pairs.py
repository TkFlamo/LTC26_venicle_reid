#!/usr/bin/env python3
import argparse
from vehicle_fingerprint.pairs import build_listwise_training, build_pair_training
p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--hard-map');p.add_argument('--out',required=True);p.add_argument('--mode',choices=['listwise','pairwise'],default='listwise');p.add_argument('--candidates-per-query',type=int,default=24);p.add_argument('--groups-per-id',type=int,default=4);p.add_argument('--knn-k',type=int,default=20);a=p.parse_args()
if a.mode=='listwise':print(build_listwise_training(a.cache,a.out,a.hard_map,candidates_per_query=a.candidates_per_query,groups_per_id=a.groups_per_id,knn_k=a.knn_k))
else:print(build_pair_training(a.cache,a.out,a.hard_map,knn_k=a.knn_k))
