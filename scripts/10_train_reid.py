#!/usr/bin/env python3
import argparse
from vehicle_fingerprint.train_reid import train_from_config

p=argparse.ArgumentParser(description='Part-aware/detail ReID training with switchable DINOv3 backbone')
p.add_argument('config')
p.add_argument('--backbone',default=None,help='Alias or direct timm model name; must match warmstart architecture')
p.add_argument('--run-dir',default=None)
p.add_argument('--warmstart',default=None)
p.add_argument('--teacher-checkpoint',default=None)
p.add_argument('--hard-negative-map',default=None)
a=p.parse_args()
print(train_from_config(
    a.config,backbone=a.backbone,run_dir=a.run_dir,warmstart=a.warmstart,
    teacher_checkpoint=a.teacher_checkpoint,hard_negative_map=a.hard_negative_map,
))
