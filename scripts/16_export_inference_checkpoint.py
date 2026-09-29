#!/usr/bin/env python3
import argparse, torch
p=argparse.ArgumentParser(); p.add_argument('--src',required=True); p.add_argument('--out',required=True)
a=p.parse_args(); ck=torch.load(a.src,map_location='cpu',weights_only=False); ck['model']={k:v for k,v in ck['model'].items() if not k.startswith('classifier.')}; ck.pop('label_map',None); torch.save(ck,a.out); print(a.out)
