#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vehicle_fingerprint.v5_checkpoint import convert_v5_checkpoint
p=argparse.ArgumentParser();p.add_argument('--src',required=True);p.add_argument('--out',required=True);a=p.parse_args()
print(convert_v5_checkpoint(a.src,a.out))
