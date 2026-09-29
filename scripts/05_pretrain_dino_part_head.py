#!/usr/bin/env python3
import argparse
from vehicle_fingerprint.train_part_head import train_part_head_from_config

p = argparse.ArgumentParser(description='Bootstrap the DINO spatial part head on Carparts-Seg')
p.add_argument('config', nargs='?', default='configs/part_head_carparts.yaml')
p.add_argument('--run-dir',default=None)
p.add_argument('--warmstart',default=None,help='Strong baseline checkpoint to protect and augment')
p.add_argument('--expect-backbone',default=None,help='Optional alias assertion, e.g. vit_base')
a = p.parse_args()
print(train_part_head_from_config(a.config,run_dir=a.run_dir,warmstart=a.warmstart,expect_backbone=a.expect_backbone))
