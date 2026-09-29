#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vehicle_fingerprint.tensorrt_deploy import build_engine_with_trtexec, build_reranker_engine_with_trtexec

def rooted(p):
    p=Path(p); return p if p.is_absolute() else ROOT/p

def main():
    ap=argparse.ArgumentParser(description='Rebuild v0.9.4 TensorRT .plan files from transferred ONNX files')
    ap.add_argument('--export-dir',default='deploy/tensorrt_v094')
    ap.add_argument('--precision',default=None,choices=['fp32','fp16','bf16'])
    ap.add_argument('--workspace-mib',type=int,default=4096)
    ap.add_argument('--trtexec',default=None)
    a=ap.parse_args(); d=rooted(a.export_dir); mf=d/'export_manifest.json'
    if not mf.is_file(): raise SystemExit(f'Missing {mf}')
    m=json.loads(mf.read_text(encoding='utf-8')); prof=m['shape_profile']; precision=a.precision or m.get('precision','fp16')
    h,w=int(prof['height']),int(prof['width']); mn=int(prof['min_batch']);op=int(prof['opt_batch']);mx=int(prof['max_batch'])
    built={}
    for name in ('vit_global','convnext_global','full_feature_extractor'):
        onnx=d/'onnx'/f'{name}.onnx'
        if not onnx.is_file():
            print(f'[SKIP] missing {onnx}'); continue
        eng=d/'engines'/f'{name}.plan'
        p,cmd=build_engine_with_trtexec(onnx,eng,input_name='input',min_shape=(mn,3,h,w),opt_shape=(op,3,h,w),max_shape=(mx,3,h,w),precision=precision,workspace_mib=a.workspace_mib,trtexec=a.trtexec)
        built[name]=str(p);print(f'[OK] {name}: {p}')
    ronnx=d/'onnx'/'pair_reranker.onnx'
    if ronnx.is_file():
        input_dim=int(m.get('models',{}).get('pair_reranker',{}).get('input_dim',33))
        p,cmd=build_reranker_engine_with_trtexec(ronnx,d/'engines'/'pair_reranker.plan',input_dim,min_pairs=1,opt_pairs=64,max_pairs=512,precision=precision,workspace_mib=min(1024,a.workspace_mib),trtexec=a.trtexec)
        built['pair_reranker']=str(p);print(f'[OK] pair_reranker: {p}')
    print(json.dumps({'precision':precision,'built':built},indent=2))

if __name__=='__main__':main()
