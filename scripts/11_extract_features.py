#!/usr/bin/env python3
import argparse
from vehicle_fingerprint.features import extract_feature_cache
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='auto');p.add_argument('--precision',default='bf16');p.add_argument('--batch',type=int,default=48);p.add_argument('--workers',type=int,default=8);p.add_argument('--image-size',type=int,nargs=2,metavar=('H','W'),default=None,help='Override checkpoint preprocessing size; normally omit')
a=p.parse_args();print(extract_feature_cache(a.manifest,a.checkpoint,a.out,device=a.device,precision=a.precision,batch_size=a.batch,workers=a.workers,image_size=a.image_size))
