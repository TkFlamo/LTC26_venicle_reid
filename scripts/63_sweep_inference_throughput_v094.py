#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def ints(raw): return [int(x.strip()) for x in str(raw).split(',') if x.strip()]

def gpu_info():
    try:
        p=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"],text=True,capture_output=True,check=False)
        return (p.stdout or "").strip().splitlines()[0] if p.returncode==0 and p.stdout.strip() else "unknown"
    except Exception:return "unknown"

def subset_input(src:Path,dst:Path,nq:int,ng:int):
    dst.mkdir(parents=True,exist_ok=True)
    for name,n in [("test_query.csv",nq),("test_gallery.csv",ng)]:
        df=pd.read_csv(src/name,dtype={"image_id":str})
        if n>0:df=df.iloc[:min(n,len(df))].copy()
        df.to_csv(dst/name,index=False)
    images=src/"images"
    target=dst/"images"
    if not target.exists(): target.symlink_to(images,target_is_directory=True)

def run(cmd,env):
    p=subprocess.run(cmd,cwd=ROOT,env=env,text=True)
    return p.returncode

def main():
    ap=argparse.ArgumentParser(description="Sweep batch/workers for PT and ONNX and report configurations that meet an image/s target")
    ap.add_argument("--input-dir",required=True)
    ap.add_argument("--deployment-dir",default="deploy/models_current")
    ap.add_argument("--onnx-dir",default="deploy/onnx_current")
    ap.add_argument("--out",default="artifacts/runtime_sweep_a5000")
    ap.add_argument("--batches",default="8,16,24,32,48,64,96")
    ap.add_argument("--workers",default="4,8,12")
    ap.add_argument("--max-query",type=int,default=512)
    ap.add_argument("--max-gallery",type=int,default=512)
    ap.add_argument("--device",default="0")
    ap.add_argument("--pt-precision",default="fp16",choices=["fp32","fp16","bf16"])
    ap.add_argument("--provider",default="cuda",choices=["auto","cuda","tensorrt","cpu"])
    ap.add_argument("--target-fps",type=float,default=100.0)
    ap.add_argument("--runtime",default="both",choices=["both","pt","onnx"])
    ap.add_argument("--warmup-batches",type=int,default=3)
    a=ap.parse_args()
    inp=Path(a.input_dir).expanduser().resolve();dep=Path(a.deployment_dir).expanduser();dep=dep if dep.is_absolute() else (ROOT/dep).resolve();onx=Path(a.onnx_dir).expanduser();onx=onx if onx.is_absolute() else (ROOT/onx).resolve();out=Path(a.out).expanduser();out=out if out.is_absolute() else (ROOT/out).resolve();out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((dep/"deployment.json").read_text(encoding="utf-8"));mode=str(meta.get("mode","single")).lower()
    env=os.environ.copy();env["PYTHONPATH"]=str(ROOT/"src")+(":"+env["PYTHONPATH"] if env.get("PYTHONPATH") else "");env["CUDA_VISIBLE_DEVICES"]=str(a.device)
    rows=[]
    with tempfile.TemporaryDirectory(prefix="v094_sweep_") as td:
        sub=Path(td)/"input";subset_input(inp,sub,a.max_query,a.max_gallery)
        nq=len(pd.read_csv(sub/"test_query.csv"));ng=len(pd.read_csv(sub/"test_gallery.csv"));n=nq+ng
        for workers in ints(a.workers):
            for batch in ints(a.batches):
                print(f"\n=== mode={mode} batch={batch} workers={workers} ===",flush=True)
                if mode=="single":
                    bd=out/f"single_b{batch}_w{workers}";shutil.rmtree(bd,ignore_errors=True)
                    cmd=[sys.executable,str(ROOT/"scripts/59_benchmark_pt_onnx_pipeline_v094.py"),"--input-dir",str(sub),"--deployment-dir",str(dep),"--onnx-dir",str(onx),"--out",str(bd),"--provider",a.provider,"--device","0","--pt-precision",a.pt_precision,"--pt-reranker-device","cuda:0","--batch",str(batch),"--workers",str(workers),"--warmup-batches",str(a.warmup_batches)]
                    rc=run(cmd,env)
                    if rc!=0:
                        rows.append({"runtime":"both","batch":batch,"workers":workers,"status":f"failed:{rc}"});continue
                    rep=json.loads((bd/"benchmark.json").read_text())
                    for runtime in ("pytorch","onnx"):
                        if a.runtime!="both" and ((a.runtime=="pt")!=(runtime=="pytorch")):continue
                        r=rep[runtime];feature=float(r["images_per_s"]);pipe=float(n/max(r["pipeline_after_manifest_s"],1e-9))
                        rows.append({"runtime":runtime,"batch":batch,"workers":workers,"feature_images_s":feature,"pipeline_images_s":pipe,"neural_query_ms_sample":r["query_feature"].get("neural_ms_per_sample"),"status":"ok"})
                elif mode=="score_ensemble":
                    runtimes=["pt","onnx"] if a.runtime=="both" else [a.runtime]
                    for runtime in runtimes:
                        rd=out/f"score_{runtime}_b{batch}_w{workers}";shutil.rmtree(rd,ignore_errors=True)
                        cmd=[sys.executable,str(ROOT/"scripts/61_infer_score_ensemble_folder_v094.py"),"--input-dir",str(sub),"--deployment-dir",str(dep),"--out",str(rd),"--runtime",runtime,"--provider",a.provider,"--device","0","--precision",a.pt_precision,"--batch",str(batch),"--workers",str(workers),"--warmup-batches",str(a.warmup_batches)]
                        if runtime=="onnx":cmd += ["--onnx-dir",str(onx)]
                        rc=run(cmd,env)
                        if rc!=0:
                            rows.append({"runtime":runtime,"batch":batch,"workers":workers,"status":f"failed:{rc}"});continue
                        rep=json.loads((rd/"score_ensemble_inference.json").read_text());pipe=float(n/max(rep["total_after_manifest_s"],1e-9))
                        rows.append({"runtime":runtime,"batch":batch,"workers":workers,"feature_images_s":rep.get("feature_images_per_s"),"pipeline_images_s":pipe,"status":"ok"})
                else: raise SystemExit(f"Unsupported deployment mode for sweep: {mode}")
                pd.DataFrame(rows).to_csv(out/"throughput_sweep.csv",index=False)
    df=pd.DataFrame(rows);df.to_csv(out/"throughput_sweep.csv",index=False)
    ok=df[df.get("status",pd.Series(dtype=str)).eq("ok")].copy() if not df.empty else df
    if not ok.empty:
        ok=ok.sort_values("pipeline_images_s",ascending=False)
        best=ok.iloc[0].to_dict();meets=ok[ok["pipeline_images_s"]>=a.target_fps]
    else:best={};meets=ok
    summary={"schema":"vehicle-reid-v094-runtime-sweep-v1","gpu":gpu_info(),"deployment_mode":mode,"target_fps":a.target_fps,"rows":len(df),"best":best,"target_met":bool(len(meets)),"output":str(out/"throughput_sweep.csv")}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\n",json.dumps(summary,ensure_ascii=False,indent=2));print(f"[OK] {out/'throughput_sweep.csv'}")

if __name__=="__main__":main()
