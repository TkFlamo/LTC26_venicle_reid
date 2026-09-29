#!/usr/bin/env python3
from __future__ import annotations
import json, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vehicle_fingerprint.tensorrt_deploy import discover_v094_global_checkpoints, discover_v094_full_deployment
try:
 import yaml; cfg=yaml.safe_load((ROOT/'configs/full_cv_pipeline.yaml').read_text())
except Exception: cfg={}
active_run_root=cfg.get("paths",{}).get("runs") if isinstance(cfg,dict) else None
info={"python":sys.version,"trtexec":shutil.which('trtexec'),"engine_builder":None,"active_run_root":active_run_root,"checkpoints":{k:str(v) for k,v in discover_v094_global_checkpoints(ROOT,active_run_root).items()},"full_deployment":{k:(str(v) if v is not None else None) for k,v in discover_v094_full_deployment(ROOT,cfg).items()}}
for name in ('torch','onnx','tensorrt','timm','modelopt'):
 try:
  m=__import__(name);info[name]=getattr(m,'__version__','installed')
 except Exception as e:info[name]=f'MISSING: {e}'

if info.get('trtexec'):
 info['engine_builder']='trtexec'
elif not str(info.get('tensorrt','')).startswith('MISSING:'):
 info['engine_builder']='python_tensorrt_builder'
else:
 info['engine_builder']='MISSING: install TensorRT Python builder or provide trtexec'
try:
 import tensorrt as _trt
 _major=int(str(_trt.__version__).split('.',1)[0])
 info['trt_major']=_major
 info['reduced_precision_requirement']='direct_typed_fp16_bf16_onnx (ModelOpt only fallback)' if _major >= 11 else 'BuilderFlag.FP16/BF16'
except Exception:
 pass
print(json.dumps(info,ensure_ascii=False,indent=2))
