#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vehicle_fingerprint.score_ensemble import read_score_ensemble_deployment


def main():
    ap=argparse.ArgumentParser(description="Export every member of a score-level ensemble to ONNX")
    ap.add_argument("--deployment-dir",default="deploy/models_current")
    ap.add_argument("--out",default="deploy/onnx_current")
    ap.add_argument("--precision",default="fp16",choices=["fp32","fp16","bf16"])
    ap.add_argument("--device",default="0")
    ap.add_argument("--opset",type=int,default=18)
    ap.add_argument("--force",action="store_true")
    a=ap.parse_args()
    dep=Path(a.deployment_dir).expanduser();dep=dep if dep.is_absolute() else (ROOT/dep).resolve()
    out=Path(a.out).expanduser();out=out if out.is_absolute() else (ROOT/out).resolve()
    meta=read_score_ensemble_deployment(dep)
    if out.exists() and a.force: shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for m in meta["members"]:
        src=Path(m["path"]); mm=json.loads((src/"deployment.json").read_text(encoding="utf-8")); mode=str(mm.get("mode","single")).lower()
        dst=out/"members"/m["name"];dst.mkdir(parents=True,exist_ok=True)
        if (dst/"onnx_manifest.json").is_file() and not a.force:
            print(f"[SKIP] {m['name']} already exported: {dst}")
        else:
            if mode=="single": script=ROOT/"scripts/57_export_single_deployment_onnx_v094.py"; flag="--deployment-dir"
            elif mode=="ensemble": script=ROOT/"scripts/52_export_mixed_ensemble_onnx_v094.py"; flag="--deployment"
            else: raise RuntimeError(f"Unsupported member mode {mode!r}: {src}")
            cmd=[sys.executable,str(script),flag,str(src),"--out",str(dst),"--precision",a.precision,"--device",str(a.device),"--opset",str(a.opset)]
            print("[RUN]"," ".join(cmd),flush=True);subprocess.run(cmd,cwd=ROOT,check=True)
        records.append({"name":m["name"],"weight":m["weight"],"deployment_mode":mode,"onnx_dir":str(Path("members")/m["name"])})
    manifest={"schema":"vehicle-reid-v094-score-ensemble-onnx-v1","precision":a.precision,"opset":a.opset,"deployment":str(dep),"fusion":meta.get("fusion"),"members":records}
    (out/"onnx_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    shutil.copy2(dep/"deployment.json",out/"deployment.json")
    print(json.dumps(manifest,ensure_ascii=False,indent=2));print(f"[OK] {out}")

if __name__=="__main__":main()
