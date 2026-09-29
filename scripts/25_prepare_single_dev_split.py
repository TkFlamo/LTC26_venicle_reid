#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vehicle_fingerprint.data.single_split import build_shared_validation_single_split_manifests

p=argparse.ArgumentParser(description='Build one fixed development train/val split plus immutable cross-project V5 shared validation')
p.add_argument('--manifest',default='data/processed/hackathon/manifest.csv')
p.add_argument('--out',default='data/processed/hackathon_single_v5shared')
p.add_argument('--fixed-validation-spec',required=True)
p.add_argument('--inner-val-fraction',type=float,default=.25)
p.add_argument('--reranker-fit-fraction-of-val',type=float,default=.50)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--min-eval-cameras',type=int,default=2)
p.add_argument('--open-set-fraction',type=float,default=.20)
p.add_argument('--max-queries-per-id',type=int,default=2)
p.add_argument('--no-brightness',action='store_true')
a=p.parse_args()
rep=build_shared_validation_single_split_manifests(
    manifest=a.manifest,
    out_dir=a.out,
    validation_spec=a.fixed_validation_spec,
    inner_val_fraction=a.inner_val_fraction,
    reranker_fit_fraction_of_val=a.reranker_fit_fraction_of_val,
    seed=a.seed,
    min_eval_cameras=a.min_eval_cameras,
    open_set_fraction=a.open_set_fraction,
    max_queries_per_id=a.max_queries_per_id,
    compute_brightness=not a.no_brightness,
)
print(json.dumps(rep,ensure_ascii=False,indent=2))
