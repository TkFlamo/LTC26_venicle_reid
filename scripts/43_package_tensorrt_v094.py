#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, platform, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def rooted(p):
    p=Path(p); return p if p.is_absolute() else ROOT/p

def main():
    ap=argparse.ArgumentParser(description="Create a portable v0.9.4 TensorRT runtime bundle")
    ap.add_argument("--export-dir",default="deploy/tensorrt_v094")
    ap.add_argument("--deployment-dir",default="deploy/models_v094_v5exact_base_veri")
    ap.add_argument("--out",default="vehicle_reid_v094_tensorrt_runtime.tar.gz")
    ap.add_argument("--include-onnx",action="store_true",help="Include large ONNX files so engines can be rebuilt on another compatible NVIDIA environment")
    args=ap.parse_args(); exp=rooted(args.export_dir); dep=rooted(args.deployment_dir); out=rooted(args.out)
    required=[exp/"export_manifest.json",exp/"engines"/"vit_global.plan",exp/"engines"/"convnext_global.plan",exp/"engines"/"full_feature_extractor.plan",exp/"engines"/"pair_reranker.plan",dep/"deployment.json",dep/"retrieval_recipe.json",dep/"refusal.json"]
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise SystemExit("Missing required files:\n"+"\n".join(missing))
    files=required[:]
    if args.include_onnx:
        files += sorted((exp/"onnx").glob("*.onnx"))
    note=(
        "Vehicle ReID v0.9.4 TensorRT runtime bundle\n"
        "Unpack at project root so paths become deploy/tensorrt_v094 and deploy/models_v094_v5exact_base_veri.\n"
        "TensorRT .plan files are not universally portable: use the same TensorRT major/version and a compatible NVIDIA GPU architecture.\n"
        "If the target machine differs, package with --include-onnx and rebuild engines there with scripts/44_rebuild_tensorrt_from_onnx_v094.py.\n"
        "Run: python scripts/42_infer_one_tensorrt_v094.py --help\n"
        "Full benchmark: python scripts/41_benchmark_tensorrt_v094.py --help\n"
    )
    tmp=exp/"TRANSFER_README.txt"; tmp.write_text(note,encoding="utf-8"); files.append(tmp)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(out,"w:gz") as tf:
        for p in files:
            if p == tmp: arc=Path("deploy/tensorrt_v094/TRANSFER_README.txt")
            elif p.is_relative_to(exp): arc=Path("deploy/tensorrt_v094")/p.relative_to(exp)
            else: arc=Path("deploy/models_v094_v5exact_base_veri")/p.relative_to(dep)
            tf.add(p,arcname=arc.as_posix(),recursive=False)
    tmp.unlink(missing_ok=True)
    print(json.dumps({"bundle":str(out),"include_onnx":bool(args.include_onnx),"files":len(files),"host":platform.platform()},indent=2))

if __name__=="__main__":main()
