#!/usr/bin/env python3
import argparse
from vehicle_fingerprint.pairs import train_reranker
p=argparse.ArgumentParser();p.add_argument('--pairs',required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=24);p.add_argument('--device',default='cuda');p.add_argument('--mode',choices=['listwise','pairwise','auto'],default='auto');a=p.parse_args();print(train_reranker(a.pairs,a.out,epochs=a.epochs,device=a.device,mode=a.mode))
