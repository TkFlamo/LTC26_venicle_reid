#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from vehicle_fingerprint.data.prepare import prepare_hackathon_dataset
from vehicle_fingerprint.data.official_protocol import build_official_protocol

p=argparse.ArgumentParser()
p.add_argument('--csv', required=True)
p.add_argument('--images', required=True)
p.add_argument('--out', required=True)
p.add_argument('--pad', type=float, default=.03)
p.add_argument('--val-fraction', type=float, default=.10, help='Identity-disjoint model-selection fraction')
p.add_argument('--eval-fraction', type=float, default=.10, help='Final untouched identity-disjoint holdout fraction')
p.add_argument('--min-eval-cameras', type=int, default=2)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--official-open-set-fraction', type=float, default=.20)
p.add_argument('--official-max-queries-per-id', type=int, default=2)
p.add_argument('--no-official-protocols', action='store_true')
a=p.parse_args()

out=Path(a.out)
df=prepare_hackathon_dataset(
    a.csv,a.images,out,pad=a.pad,val_fraction=a.val_fraction,eval_fraction=a.eval_fraction,
    min_eval_cameras=a.min_eval_cameras,seed=a.seed
)
print(df.groupby('split').size())
print('IDs by split:', df.groupby('split').vehicle_id.nunique().to_dict())
rp=out/'split_report.json'
if rp.exists():
    print(json.dumps(json.loads(rp.read_text(encoding='utf-8')),ensure_ascii=False,indent=2))

if not a.no_official_protocols:
    for split, seed_offset in [('val',0),('eval',10000)]:
        mf=out/f'{split}.csv'
        if mf.exists():
            report=build_official_protocol(
                mf,out/f'official_{split}',seed=a.seed+seed_offset,
                open_set_fraction=a.official_open_set_fraction,
                max_queries_per_id=a.official_max_queries_per_id,
            )
            print(f'official_{split}:')
            print(json.dumps(report,ensure_ascii=False,indent=2))
