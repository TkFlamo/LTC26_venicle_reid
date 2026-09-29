#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.data.prepare import prepare_hackathon_dataset
from vehicle_fingerprint.runtime_onnx import (
    ONNXRerankerBackend,
    extract_feature_cache_onnx_timed,
    read_onnx_manifest,
    run_retrieval_backend_timed,
)


def main():
    p = argparse.ArgumentParser(description="Complete single-deployment ONNX inference for test_query/test_gallery/images folder")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--onnx-dir", default="deploy/onnx_current")
    p.add_argument("--out", default="outputs/hackathon_test_onnx")
    p.add_argument("--query-name", default="test_query.csv")
    p.add_argument("--gallery-name", default="test_gallery.csv")
    p.add_argument("--images-name", default="images")
    p.add_argument("--provider", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--warmup-batches", type=int, default=1)
    p.add_argument("--query-id-column", default=None)
    p.add_argument("--gallery-id-column", default=None)
    a = p.parse_args()

    inp = Path(a.input_dir).expanduser().resolve()
    onnx_dir = Path(a.onnx_dir).expanduser(); onnx_dir = onnx_dir if onnx_dir.is_absolute() else (ROOT / onnx_dir).resolve()
    out = Path(a.out).expanduser(); out = out if out.is_absolute() else (ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True); work = out / "work"; work.mkdir(parents=True, exist_ok=True)

    qcsv, gcsv, images = inp / a.query_name, inp / a.gallery_name, inp / a.images_name
    for pth in (qcsv, gcsv):
        if not pth.is_file(): raise SystemExit(f"Missing: {pth}")
    if not images.is_dir(): raise SystemExit(f"Missing images directory: {images}")

    model_onnx = onnx_dir / "full_feature_extractor.onnx"
    if not model_onnx.is_file(): raise SystemExit(f"Missing: {model_onnx}")
    manifest = read_onnx_manifest(onnx_dir)
    dep_json = onnx_dir / "deployment.json"
    if dep_json.is_file():
        dep = json.loads(dep_json.read_text(encoding="utf-8"))
        if str(dep.get("mode", "single")).lower() != "single":
            raise SystemExit("58_infer_onnx_folder_v094.py currently expects a single deployment")
    recipe_path = onnx_dir / "retrieval_recipe.json"; refusal_path = onnx_dir / "refusal.json"
    if not recipe_path.is_file() or not refusal_path.is_file():
        raise SystemExit("ONNX directory must contain retrieval_recipe.json and refusal.json")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8")); refusal = json.loads(refusal_path.read_text(encoding="utf-8"))

    times = {}
    t = time.perf_counter()
    qdir = work / "query"; prepare_hackathon_dataset(qcsv, images, qdir, pad=.03, val_fraction=.0, eval_fraction=.0, materialize_crops=False)
    gdir = work / "gallery"; prepare_hackathon_dataset(gcsv, images, gdir, pad=.03, val_fraction=.0, eval_fraction=.0, materialize_crops=False)
    times["manifest_prepare_s"] = time.perf_counter() - t
    qmanifest = qdir / "test.csv" if (qdir/"test.csv").is_file() else qdir/"manifest.csv"
    gmanifest = gdir / "test.csv" if (gdir/"test.csv").is_file() else gdir/"manifest.csv"

    qcache = work / "query_features_onnx.npz"; gcache = work / "gallery_features_onnx.npz"
    _, qstats = extract_feature_cache_onnx_timed(qmanifest, model_onnx, qcache, onnx_manifest=manifest, provider=a.provider, device=a.device, batch_size=a.batch, workers=a.workers, warmup_batches=a.warmup_batches)
    _, gstats = extract_feature_cache_onnx_timed(gmanifest, model_onnx, gcache, onnx_manifest=manifest, provider=a.provider, device=a.device, batch_size=a.batch, workers=a.workers, warmup_batches=a.warmup_batches)

    rr_path = onnx_dir / "pair_reranker.onnx"
    rr = ONNXRerankerBackend(rr_path, provider=a.provider, device=a.device) if rr_path.is_file() and float(recipe.get("reranker_beta",0)) > 0 else None
    result = run_retrieval_backend_timed(qcache, gcache, out, recipe=recipe, refusal=refusal, reranker_backend=rr, query_id_column=a.query_id_column, gallery_id_column=a.gallery_id_column, write=True)

    report = {
        "runtime": "onnxruntime",
        "onnx_dir": str(onnx_dir),
        "provider": qstats.get("provider"),
        "query_feature": qstats,
        "gallery_feature": gstats,
        "reranker_init_s": float(rr.init_s) if rr else 0.0,
        "retrieval": result["timings"],
        "manifest_prepare_s": times["manifest_prepare_s"],
        "total_feature_samples": int(qstats["samples"] + gstats["samples"]),
    }
    report["total_measured_s"] = float(times["manifest_prepare_s"] + qstats["total_s"] + gstats["total_s"] + (rr.init_s if rr else 0.0) + result["timings"]["total_retrieval_s"])
    (out / "onnx_inference_timing.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] {out/'submission.csv'}")


if __name__ == "__main__":
    main()
