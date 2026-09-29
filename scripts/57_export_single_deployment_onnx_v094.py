#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.features import load_inference_model
from vehicle_fingerprint.onnx_export import (
    FULL_OUTPUTS,
    FullFeatureWrapper,
    torch_dtype_for_precision,
    export_onnx,
    export_reranker_onnx,
    sha256_file,
)


def check_onnx(path: Path) -> dict:
    import onnx
    m = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(m)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "outputs": list(FULL_OUTPUTS) if "full_feature" in path.name else None,
    }


def resolve(dep: Path, value, fallback):
    p = Path(value or fallback).expanduser()
    return (dep / p).resolve() if not p.is_absolute() else p.resolve()


def main():
    p = argparse.ArgumentParser(description="Export a SINGLE v0.9.4 deployment to ONNX")
    p.add_argument("--deployment-dir", default="deploy/models_current")
    p.add_argument("--out", default="deploy/onnx_current")
    p.add_argument("--precision", default="fp32", choices=["fp32","fp16"])
    p.add_argument("--device", default="0")
    p.add_argument("--opset", type=int, default=18)
    a = p.parse_args()

    dep = Path(a.deployment_dir).expanduser()
    if not dep.is_absolute(): dep = (ROOT/dep).resolve()
    out = Path(a.out).expanduser()
    if not out.is_absolute(): out = (ROOT/out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((dep/"deployment.json").read_text(encoding="utf-8"))
    mode = str(meta.get("mode", "single")).lower()
    if mode != "single":
        raise SystemExit(
            "This exporter is for a single deployment. For an ensemble use "
            "scripts/62_export_production_ensemble_onnx_v094.py."
        )

    ck = resolve(dep, meta.get("checkpoint"), "reid.pt")
    rr = resolve(dep, meta.get("reranker"), "reranker.pt")
    if not ck.is_file():
        raise FileNotFoundError(ck)

    if a.precision == "fp32":
        dev = torch.device("cpu")
        dtype = torch.float32
        model_dtype = None
    else:
        if not torch.cuda.is_available():
            raise SystemExit(f"{a.precision} export requires CUDA")
        dev = torch.device(f"cuda:{a.device}")
        dtype = torch_dtype_for_precision(a.precision)
        model_dtype = dtype

    model, _, mcfg = load_inference_model(ck, device=str(dev))
    size = tuple(map(int, (mcfg.get("preprocess",{}) or {}).get("image_size",[384,576])))
    wrapper = FullFeatureWrapper(model, size).eval()
    full = out/"full_feature_extractor.onnx"
    export_onnx(
        wrapper, full, image_size=size, output_names=FULL_OUTPUTS,
        opset=a.opset, input_dtype=dtype, export_device=dev, model_dtype=model_dtype,
    )

    manifest = {
        "schema": "vehicle-reid-v094-single-onnx-v1",
        "deployment": str(dep),
        "precision": a.precision,
        "opset": a.opset,
        "image_size": list(size),
        "preprocess": mcfg.get("preprocess", {}) or {},
        "models": {
            "full_feature_extractor": {
                **check_onnx(full),
                "checkpoint": str(ck),
                "checkpoint_sha256": sha256_file(ck),
            }
        },
        "runtime_note": (
            "Neural feature extractor and optional pair reranker are ONNX. "
            "Retrieval recipe, k-reciprocal logic and refusal remain runtime logic."
        ),
    }

    if rr.is_file():
        rr_out = out/"pair_reranker.onnx"
        rp, dim = export_reranker_onnx(
            rr, rr_out, opset=a.opset, input_dtype=dtype,
            export_device=dev, model_dtype=model_dtype,
        )
        manifest["models"]["pair_reranker"] = {
            **check_onnx(rp),
            "checkpoint": str(rr),
            "checkpoint_sha256": sha256_file(rr),
            "input_dim": int(dim),
        }

    # Keep runtime metadata beside ONNX.
    for name in ("deployment.json","retrieval_recipe.json","refusal.json"):
        src = dep/name
        if src.is_file():
            shutil.copy2(src, out/name)

    (out/"onnx_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
