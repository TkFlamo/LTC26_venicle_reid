#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TOP={'.git','.venv','venv','data','runs','artifacts','weights','outputs','__pycache__'};NAMES={'.pytest_cache','.mypy_cache','.ruff_cache','.ipynb_checkpoints','__pycache__'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while True:
   b=f.read(8*1024*1024)
   if not b:break
   h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--deployment',default='deploy/models_v094_mixed_ensemble');p.add_argument('--onnx',default='deploy/onnx_v094_mixed_ensemble');p.add_argument('--out',default='../vehicle_fingerprint_reid_v094_clean_mixed_ensemble.zip');a=p.parse_args();dep=Path(a.deployment);onn=Path(a.onnx);out=Path(a.out).expanduser();out=out if out.is_absolute() else (ROOT/out).resolve();out.parent.mkdir(parents=True,exist_ok=True)
 if not (ROOT/dep/'deployment.json').is_file():raise SystemExit(f'Missing {ROOT/dep}')
 if not (ROOT/onn/'onnx_manifest.json').is_file():raise SystemExit(f'Missing {ROOT/onn}')
 def ok(rel):
  if not rel.parts or rel.parts[0] in TOP or any(x in NAMES for x in rel.parts) or rel.suffix.lower() in {'.pyc','.pyo'}:return False
  if rel.parts[0]=='deploy':return rel.is_relative_to(dep) or rel.is_relative_to(onn)
  return True
 files=[(f.relative_to(ROOT),f) for f in ROOT.rglob('*') if f.is_file() and ok(f.relative_to(ROOT))];rn='vehicle_fingerprint_reid_v094_clean_mixed_ensemble';mf={'schema':'vehicle-reid-v094-clean-mixed-ensemble-package-v1','deployment':str(dep),'onnx':str(onn),'files':{}}
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
  for rel,f in sorted(files,key=lambda x:str(x[0])):z.write(f,(Path(rn)/rel).as_posix());mf['files'][rel.as_posix()]={'bytes':f.stat().st_size,'sha256':sha(f)}
  z.writestr((Path(rn)/'PACKAGE_MANIFEST.json').as_posix(),json.dumps(mf,ensure_ascii=False,indent=2))
 d=sha(out);sp=out.with_name(out.name+'_SHA256.txt');sp.write_text(f'{d}  {out.name}\n');print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'files':len(files),'sha256':d,'sha256_file':str(sp)},indent=2))
if __name__=='__main__':main()
