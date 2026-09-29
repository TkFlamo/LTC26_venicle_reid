#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vehicle_fingerprint.baseline_v5_exact import train_v5_exact_presplit


def main():
    p=argparse.ArgumentParser(description='Exact vehicle_reid_v5_official global trainer on a fixed pre-split development set')
    p.add_argument('--train-manifest',required=True)
    p.add_argument('--val-manifest',default=None)
    p.add_argument('--images-dir',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--profile',default='base',choices=['small','base','large'])
    p.add_argument('--height',type=int,default=384)
    p.add_argument('--width',type=int,default=576)
    p.add_argument('--device',default='0')
    p.add_argument('--precision',default='bf16',choices=['fp32','fp16','bf16'])
    p.add_argument('--epochs',type=int,default=None)
    p.add_argument('--train-microbatch',type=int,default=None)
    p.add_argument('--init-checkpoint',default=None,help='Native V5 checkpoint used as transfer initialization; classifier is skipped automatically when identity maps differ.')
    p.add_argument('--plate-mask-prob',type=float,default=None,help='Optional randomized plausible plate-region degradation probability; useful for VeRi pretraining ablations.')
    p.add_argument('--final-refit',action='store_true')
    a=p.parse_args()
    ck=train_v5_exact_presplit(train_manifest=a.train_manifest,val_manifest=a.val_manifest,images_dir=a.images_dir,out_dir=a.out,model_name=a.model,profile=a.profile,height=a.height,width=a.width,device=a.device,precision=a.precision,epochs=a.epochs,final_refit=a.final_refit,train_microbatch=a.train_microbatch,init_checkpoint=a.init_checkpoint,plate_mask_prob=a.plate_mask_prob)
    print(ck)

if __name__=='__main__': main()
