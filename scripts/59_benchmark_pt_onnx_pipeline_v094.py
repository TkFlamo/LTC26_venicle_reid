#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.data.prepare import prepare_hackathon_dataset
from vehicle_fingerprint.runtime_onnx import (
    ONNXRerankerBackend,
    PTRerankerBackend,
    cache_equivalence,
    extract_feature_cache_onnx_timed,
    extract_feature_cache_pt_timed,
    ranking_agreement,
    read_onnx_manifest,
    run_retrieval_backend_timed,
)


def resolve(dep: Path, value, fallback):
    p = Path(value or fallback).expanduser()
    return (dep / p).resolve() if not p.is_absolute() else p.resolve()


def flatten(prefix: str, x: dict, out: dict):
    for k, v in x.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict): flatten(key, v, out)
        elif isinstance(v, (str, int, float, bool)) or v is None: out[key] = v


def subset_csv(src: Path, dst: Path, n: int | None):
    df = pd.read_csv(src, dtype={"image_id": str})
    if n is not None and n > 0: df = df.iloc[:min(int(n), len(df))].copy()
    df.to_csv(dst, index=False)
    return len(df)


def main():
    p = argparse.ArgumentParser(description="Stage-by-stage PT vs ONNX benchmark for a single v0.9.4 deployment")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--deployment-dir", default="deploy/models_current")
    p.add_argument("--onnx-dir", default="deploy/onnx_current")
    p.add_argument("--out", default="artifacts/benchmark_pt_onnx")
    p.add_argument("--provider", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--device", default="0")
    p.add_argument("--pt-precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--pt-reranker-device", default="cpu", help="cpu matches current scripts/21 inference; use cuda:0 for GPU microbenchmark")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--workers", type=int, default=0, help="0 gives the cleanest preprocess timing; use 8 for realistic wall throughput")
    p.add_argument("--warmup-batches", type=int, default=1)
    p.add_argument("--max-query", type=int, default=0, help="0 = all")
    p.add_argument("--max-gallery", type=int, default=0, help="0 = all")
    p.add_argument("--write-inference-artifacts", action="store_true")
    a = p.parse_args()

    inp = Path(a.input_dir).expanduser().resolve()
    dep = Path(a.deployment_dir).expanduser(); dep = dep if dep.is_absolute() else (ROOT/dep).resolve()
    onx = Path(a.onnx_dir).expanduser(); onx = onx if onx.is_absolute() else (ROOT/onx).resolve()
    out = Path(a.out).expanduser(); out = out if out.is_absolute() else (ROOT/out).resolve(); out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((dep/"deployment.json").read_text(encoding="utf-8"))
    if str(meta.get("mode", "single")).lower() != "single": raise SystemExit("Benchmark 59 currently expects a single deployment")
    ck = resolve(dep, meta.get("checkpoint"), "reid.pt")
    rr_pt_path = resolve(dep, meta.get("reranker"), "reranker.pt")
    recipe = json.loads((dep/"retrieval_recipe.json").read_text(encoding="utf-8"))
    refusal = json.loads((dep/"refusal.json").read_text(encoding="utf-8"))
    full_onnx = onx/"full_feature_extractor.onnx"; rr_onnx = onx/"pair_reranker.onnx"
    if not ck.is_file() or not full_onnx.is_file(): raise SystemExit(f"Missing PT/ONNX model: {ck}, {full_onnx}")

    qsrc, gsrc, images = inp/"test_query.csv", inp/"test_gallery.csv", inp/"images"
    if not qsrc.is_file() or not gsrc.is_file() or not images.is_dir(): raise SystemExit("input-dir must contain test_query.csv, test_gallery.csv, images/")

    with tempfile.TemporaryDirectory(prefix="v094_bench_") as td0:
        td = Path(td0)
        qcsv, gcsv = td/"test_query.csv", td/"test_gallery.csv"
        nq = subset_csv(qsrc, qcsv, a.max_query or None); ng = subset_csv(gsrc, gcsv, a.max_gallery or None)
        t = time.perf_counter()
        qprep = td/"query"; gprep = td/"gallery"
        prepare_hackathon_dataset(qcsv, images, qprep, pad=.03, val_fraction=.0, eval_fraction=.0, materialize_crops=False)
        prepare_hackathon_dataset(gcsv, images, gprep, pad=.03, val_fraction=.0, eval_fraction=.0, materialize_crops=False)
        manifest_prepare_s = time.perf_counter() - t
        qm = qprep/"test.csv" if (qprep/"test.csv").is_file() else qprep/"manifest.csv"
        gm = gprep/"test.csv" if (gprep/"test.csv").is_file() else gprep/"manifest.csv"

        ptq_path, ptg_path = td/"pt_q.npz", td/"pt_g.npz"
        oxq_path, oxg_path = td/"ox_q.npz", td/"ox_g.npz"
        ptq, ptq_t = extract_feature_cache_pt_timed(qm, ck, ptq_path, device=a.device, precision=a.pt_precision, batch_size=a.batch, workers=a.workers, warmup_batches=a.warmup_batches)
        ptg, ptg_t = extract_feature_cache_pt_timed(gm, ck, ptg_path, device=a.device, precision=a.pt_precision, batch_size=a.batch, workers=a.workers, warmup_batches=a.warmup_batches)
        omf = read_onnx_manifest(onx)
        oxq, oxq_t = extract_feature_cache_onnx_timed(qm, full_onnx, oxq_path, onnx_manifest=omf, provider=a.provider, device=a.device, batch_size=a.batch, workers=a.workers, warmup_batches=a.warmup_batches)
        oxg, oxg_t = extract_feature_cache_onnx_timed(gm, full_onnx, oxg_path, onnx_manifest=omf, provider=a.provider, device=a.device, batch_size=a.batch, workers=a.workers, warmup_batches=a.warmup_batches)

        pt_rr = PTRerankerBackend(rr_pt_path, device=a.pt_reranker_device) if rr_pt_path.is_file() and float(recipe.get("reranker_beta",0)) > 0 else None
        ox_rr = ONNXRerankerBackend(rr_onnx, provider=a.provider, device=a.device) if rr_onnx.is_file() and float(recipe.get("reranker_beta",0)) > 0 else None

        pt_out = out/"pt" if a.write_inference_artifacts else td/"pt_out"
        ox_out = out/"onnx" if a.write_inference_artifacts else td/"ox_out"
        ptr = run_retrieval_backend_timed(ptq, ptg, pt_out, recipe=recipe, refusal=refusal, reranker_backend=pt_rr, write=a.write_inference_artifacts)
        oxr = run_retrieval_backend_timed(oxq, oxg, ox_out, recipe=recipe, refusal=refusal, reranker_backend=ox_rr, write=a.write_inference_artifacts)

        report = {
            "schema": "vehicle-reid-v094-pt-onnx-stage-benchmark-v1",
            "input": {"query_rows": nq, "gallery_rows": ng, "manifest_prepare_s": manifest_prepare_s},
            "config": {"batch": a.batch, "workers": a.workers, "pt_precision": a.pt_precision, "onnx_provider": oxq_t.get("provider"), "pt_reranker_device": a.pt_reranker_device},
            "pytorch": {"query_feature": ptq_t, "gallery_feature": ptg_t, "reranker_init_s": float(pt_rr.init_s) if pt_rr else 0.0, "retrieval": ptr["timings"]},
            "onnx": {"query_feature": oxq_t, "gallery_feature": oxg_t, "reranker_init_s": float(ox_rr.init_s) if ox_rr else 0.0, "retrieval": oxr["timings"]},
            "equivalence": {
                "query_cache": cache_equivalence(ptq, oxq),
                "gallery_cache": cache_equivalence(ptg, oxg),
                "ranking": ranking_agreement(ptr["ranked"], oxr["ranked"], topk=10),
            },
        }
        for mode in ("pytorch", "onnx"):
            z = report[mode]
            z["feature_total_s"] = float(z["query_feature"]["total_s"] + z["gallery_feature"]["total_s"])
            z["pipeline_after_manifest_s"] = float(z["feature_total_s"] + z["reranker_init_s"] + z["retrieval"]["total_retrieval_s"])
            z["images_per_s"] = float((nq + ng) / max(z["feature_total_s"], 1e-9))
        report["speedup_pt_over_onnx"] = {
            "feature_total_x": float(report["pytorch"]["feature_total_s"] / max(report["onnx"]["feature_total_s"], 1e-9)),
            "pipeline_after_manifest_x": float(report["pytorch"]["pipeline_after_manifest_s"] / max(report["onnx"]["pipeline_after_manifest_s"], 1e-9)),
            "feature_neural_forward_x": float(
                (report["pytorch"]["query_feature"]["neural_forward_s"] + report["pytorch"]["gallery_feature"]["neural_forward_s"]) /
                max(report["onnx"]["query_feature"]["neural_forward_s"] + report["onnx"]["gallery_feature"]["neural_forward_s"], 1e-9)
            ),
            "reranker_neural_x": float(report["pytorch"]["retrieval"].get("reranker_neural_s", 0.0) / max(report["onnx"]["retrieval"].get("reranker_neural_s", 0.0), 1e-9)) if report["onnx"]["retrieval"].get("reranker_neural_s", 0.0) > 0 else None,
        }

        (out/"benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        flat = {}; flatten("", report, flat); pd.DataFrame([flat]).to_csv(out/"benchmark_summary.csv", index=False)

        rows = []
        for mode in ("pytorch", "onnx"):
            fsum = report[mode]
            for split in ("query_feature", "gallery_feature"):
                for k, v in fsum[split].items():
                    if isinstance(v, (int,float)) and (k.endswith("_s") or k.endswith("_ms_per_sample") or k.endswith("_samples_s")):
                        rows.append({"runtime": mode, "stage": f"{split}.{k}", "seconds_or_value": float(v)})
            for k,v in fsum["retrieval"].items():
                if isinstance(v,(int,float)) and k.endswith("_s"):
                    rows.append({"runtime": mode, "stage": f"retrieval.{k}", "seconds_or_value": float(v)})
        pd.DataFrame(rows).to_csv(out/"stage_timings.csv", index=False)

        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"[OK] {out/'benchmark.json'}")
        print(f"[OK] {out/'stage_timings.csv'}")


if __name__ == "__main__":
    main()
