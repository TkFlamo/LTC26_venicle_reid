#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from vehicle_fingerprint.hard_mining import mine_identity_hard_negatives
p=argparse.ArgumentParser()
p.add_argument('--cache',required=True); p.add_argument('--out',required=True); p.add_argument('--topk',type=int,default=40)
p.add_argument('--representation',choices=['global','fused','blend'],default='blend')
p.add_argument('--alpha',type=float,default=0.25)
p.add_argument('--recipe',help='Optional retrieval_recipe.json; its base_alpha overrides --alpha for blend mining')
p.add_argument('--refine-factor',type=int,default=4); p.add_argument('--top-pair-mean',type=int,default=3)
a=p.parse_args()
alpha=a.alpha
if a.recipe:
    alpha=float(json.loads(Path(a.recipe).read_text(encoding='utf-8')).get('base_alpha',alpha))
x=mine_identity_hard_negatives(a.cache,a.out,a.topk,a.representation,alpha,a.refine_factor,a.top_pair_mean)
print('ids',len(x),'representation',a.representation,'alpha',alpha)
