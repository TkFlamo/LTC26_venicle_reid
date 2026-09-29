#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vehicle_fingerprint.features import load_inference_model
from vehicle_fingerprint.onnx_export import FULL_OUTPUTS,FullFeatureWrapper,torch_dtype_for_precision,export_onnx,export_reranker_onnx,sha256_file

def chk(path):
 import onnx
 m=onnx.load(str(path));onnx.checker.check_model(m)
 return {'path':str(path),'sha256':sha256_file(path),'size_bytes':path.stat().st_size,'ir_version':int(m.ir_version),'opsets':[{'domain':x.domain,'version':int(x.version)} for x in m.opset_import]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--deployment',default='deploy/models_v094_mixed_ensemble');p.add_argument('--out',default='deploy/onnx_v094_mixed_ensemble');p.add_argument('--precision',default='fp32',choices=['fp32','fp16','bf16']);p.add_argument('--device',default='0');p.add_argument('--opset',type=int,default=18);a=p.parse_args()
 dep=Path(a.deployment).expanduser();dep=dep if dep.is_absolute() else ROOT/dep;out=Path(a.out).expanduser();out=out if out.is_absolute() else ROOT/out;out.mkdir(parents=True,exist_ok=True)
 meta=json.loads((dep/'deployment.json').read_text());
 if str(meta.get('mode'))!='ensemble':raise SystemExit('deployment.json mode must be ensemble')
 if a.precision=='fp32':dev=torch.device('cpu');dtype=torch.float32;model_dtype=None
 else:
  if not torch.cuda.is_available():raise SystemExit(f'{a.precision} export requires CUDA')
  dev=torch.device(f'cuda:{a.device}');dtype=torch_dtype_for_precision(a.precision);model_dtype=dtype
 manifest={'schema':'vehicle-reid-v094-mixed-ensemble-onnx-v1','precision':a.precision,'opset':a.opset,'deployment':str(dep),'ensemble':{'weight_a':meta.get('weight_a'),'convnext_weight':meta.get('convnext_weight'),'vit_weight':meta.get('vit_weight')},'models':{},'runtime_note':'Two feature extractors and pair reranker are ONNX. Weighted fusion, top-k/k-reciprocal and refusal remain non-neural runtime logic.'}
 items=[('convnext_small',dep/meta['checkpoint_a'],out/'convnext_small_full.onnx'),('vit_base',dep/meta['checkpoint_b'],out/'vit_base_full.onnx')]
 for name,ck,onx in items:
  model,_,mcfg=load_inference_model(ck,device=str(dev));size=tuple(map(int,mcfg.get('preprocess',{}).get('image_size',[384,576])));wrapper=FullFeatureWrapper(model,size).eval();export_onnx(wrapper,onx,image_size=size,output_names=FULL_OUTPUTS,opset=a.opset,input_dtype=dtype,export_device=dev,model_dtype=model_dtype);r=chk(onx);r.update({'checkpoint':str(ck),'checkpoint_sha256':sha256_file(ck),'image_size':list(size),'outputs':list(FULL_OUTPUTS)});manifest['models'][name]=r
 rr=dep/meta.get('reranker','reranker.pt')
 if rr.is_file():
  rp,dim=export_reranker_onnx(rr,out/'pair_reranker.onnx',opset=a.opset,input_dtype=dtype,export_device=dev,model_dtype=model_dtype);r=chk(rp);r.update({'checkpoint':str(rr),'checkpoint_sha256':sha256_file(rr),'input_dim':int(dim)});manifest['models']['pair_reranker']=r
 for n in ('deployment.json','retrieval_recipe.json','refusal.json'):
  s=dep/n
  if s.is_file():(out/n).write_bytes(s.read_bytes())
 (out/'onnx_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
