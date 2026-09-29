#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.data.prepare import prepare_hackathon_dataset
from vehicle_fingerprint.official_eval import blended_embedding
from vehicle_fingerprint.runtime_onnx import (
    FULL_OUTPUTS,
    _make_dataset_loader,
    _ort_numpy_dtype,
    create_ort_session,
    read_onnx_manifest,
)
from vehicle_fingerprint.score_ensemble import read_score_ensemble_deployment, weighted_concat_embeddings

EXPECTED = {"base_full": 0.65, "external_convnext_full": 0.35}


def _sync(device: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(int(device))


def _validate(meta: dict) -> None:
    got = {m["name"]: float(m["weight"]) for m in meta["members"]}
    if set(got) != set(EXPECTED) or any(abs(got[k] - v) > 1e-9 for k, v in EXPECTED.items()):
        raise RuntimeError(f"Expected fixed production ensemble {EXPECTED}; got {got}")


def _build_sessions(meta: dict, onnx_root: Path, provider: str, device: str):
    sessions = []
    preprocess_sigs = []
    for member in meta["members"]:
        name = member["name"]
        odir = onnx_root / "members" / name
        mf = read_onnx_manifest(odir)
        onx = odir / "full_feature_extractor.onnx"
        if not onx.is_file():
            raise FileNotFoundError(onx)
        sess, chosen, available = create_ort_session(onx, provider=provider, device=device)
        inp = sess.get_inputs()[0]
        outs = [x.name for x in sess.get_outputs()]
        missing = [x for x in FULL_OUTPUTS if x not in outs]
        if missing:
            raise RuntimeError(f"{name}: missing outputs {missing}")
        dep = Path(member["path"])
        dm = json.loads((dep / "deployment.json").read_text(encoding="utf-8"))
        recipe_path = dep / str(dm.get("retrieval_recipe", "retrieval_recipe.json"))
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        size = tuple(map(int, mf.get("image_size") or [384, 576]))
        prep = dict(mf.get("preprocess", {}) or {})
        prep.setdefault("augmentation_profile", "baseline_v4")
        prep.setdefault("use_source_bbox", True)
        prep.setdefault("bbox_pad", .03)
        preprocess_sigs.append((size, prep.get("augmentation_profile"), prep.get("use_source_bbox"), prep.get("bbox_pad"), inp.type))
        sessions.append({
            "name": name,
            "weight": float(member["weight"]),
            "session": sess,
            "input_name": inp.name,
            "input_dtype": _ort_numpy_dtype(inp.type),
            "input_type": inp.type,
            "image_size": size,
            "prep": prep,
            "base_alpha": float(recipe.get("base_alpha", 0.0)),
            "provider": chosen,
            "available_providers": available,
        })
    if len(set(preprocess_sigs)) != 1:
        raise RuntimeError(f"Member preprocessing differs; cannot benchmark shared production extract(): {preprocess_sigs}")
    return sessions


def _batch_embedding(batch: dict, sessions: list[dict]) -> np.ndarray:
    x = np.ascontiguousarray(batch["image"].numpy().astype(sessions[0]["input_dtype"], copy=False))
    member_emb = []
    weights = []
    for info in sessions:
        outs = info["session"].run(list(FULL_OUTPUTS), {info["input_name"]: x})
        o = {k: v for k, v in zip(FULL_OUTPUTS, outs)}
        emb = blended_embedding(np.asarray(o["z_global"], np.float32), np.asarray(o["z_fused"], np.float32), info["base_alpha"])
        member_emb.append(emb)
        weights.append(info["weight"])
    fused = weighted_concat_embeddings(member_emb, weights)
    # Touch the shared color descriptor as part of extract() postprocessing.
    if "color" in batch:
        _ = batch["color"].numpy()
    return fused


def _run_batch(batch: dict, sessions: list[dict], device: str) -> int:
    emb = _batch_embedding(batch, sessions)
    return int(emb.shape[0])


def _next_cycling(loader, state):
    try:
        return next(state[0])
    except StopIteration:
        state[0] = iter(loader)
        return next(state[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Organizer-style extract() speed benchmark for the fixed dual-ONNX production ensemble")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--deployment-dir", default="deploy/models_current")
    ap.add_argument("--onnx-dir", default="deploy/onnx_current")
    ap.add_argument("--out", default="artifacts/official_extract_benchmark")
    ap.add_argument("--provider", choices=["cuda", "cpu", "auto"], default="cuda")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--latency-warmup", type=int, default=50)
    ap.add_argument("--latency-runs", type=int, default=300)
    ap.add_argument("--throughput-seconds", type=float, default=10.0)
    a = ap.parse_args()

    inp = Path(a.input_dir).expanduser().resolve()
    dep = Path(a.deployment_dir).expanduser()
    if not dep.is_absolute(): dep = (ROOT / dep).resolve()
    onnx_root = Path(a.onnx_dir).expanduser()
    if not onnx_root.is_absolute(): onnx_root = (ROOT / onnx_root).resolve()
    out = Path(a.out).expanduser()
    if not out.is_absolute(): out = (ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    meta = read_score_ensemble_deployment(dep)
    _validate(meta)
    t_load = time.perf_counter()
    sessions = _build_sessions(meta, onnx_root, a.provider, a.device)
    model_load_s = time.perf_counter() - t_load
    onnx_weight_bytes = sum(p.stat().st_size for p in onnx_root.rglob("*.onnx") if p.is_file())

    q = pd.read_csv(inp / "test_query.csv", dtype={"image_id": str})
    g = pd.read_csv(inp / "test_gallery.csv", dtype={"image_id": str})
    combined = pd.concat([q, g], ignore_index=True)
    with tempfile.TemporaryDirectory(prefix="v094_official_bench_") as td:
        td = Path(td)
        csv_path = td / "objects.csv"
        combined.to_csv(csv_path, index=False)
        prep_dir = td / "prepared"
        prepare_hackathon_dataset(csv_path, inp / "images", prep_dir, pad=.03, val_fraction=.0, eval_fraction=.0, materialize_crops=False)
        manifest = prep_dir / ("test.csv" if (prep_dir / "test.csv").is_file() else "manifest.csv")

        size, prep = sessions[0]["image_size"], sessions[0]["prep"]

        # Organizer also records determinism. Re-run the exact same preprocessed batch twice.
        _, det_loader = _make_dataset_loader(manifest, size, prep, batch_size=8, workers=0)
        det_batch = next(iter(det_loader))
        _sync(a.device)
        det_a = _batch_embedding(det_batch, sessions)
        _sync(a.device)
        det_b = _batch_embedding(det_batch, sessions)
        _sync(a.device)
        determinism_max_abs = float(np.max(np.abs(det_a - det_b)))
        deterministic = bool(np.array_equal(det_a, det_b) or np.allclose(det_a, det_b, rtol=0.0, atol=1e-6))

        # Official latency_b1: full per-object path, 50 warmups + 300 measured runs.
        _, latency_loader = _make_dataset_loader(manifest, size, prep, batch_size=1, workers=0)
        state = [iter(latency_loader)]
        for _ in range(int(a.latency_warmup)):
            b = _next_cycling(latency_loader, state)
            _sync(a.device)
            _run_batch(b, sessions, a.device)
            _sync(a.device)
        latency_ms = []
        for _ in range(int(a.latency_runs)):
            _sync(a.device)
            t = time.perf_counter()
            b = _next_cycling(latency_loader, state)
            _run_batch(b, sessions, a.device)
            _sync(a.device)
            latency_ms.append((time.perf_counter() - t) * 1000.0)

        throughput = []
        for batch_size in (1, 8, 16, 32):
            _, loader = _make_dataset_loader(manifest, size, prep, batch_size=batch_size, workers=a.workers)
            state = [iter(loader)]
            # Warm the loader/sessions at this batch size; warmup is outside the timed interval.
            for _ in range(3):
                b = _next_cycling(loader, state)
                _run_batch(b, sessions, a.device)
            _sync(a.device)
            start = time.perf_counter()
            n = 0
            while True:
                b = _next_cycling(loader, state)
                n += _run_batch(b, sessions, a.device)
                now = time.perf_counter()
                if now - start >= float(a.throughput_seconds):
                    _sync(a.device)
                    elapsed = time.perf_counter() - start
                    break
            throughput.append({"batch": batch_size, "samples": n, "seconds": elapsed, "fps": n / elapsed})

    best = max(throughput, key=lambda x: x["fps"])
    report = {
        "schema": "vehicle-reid-v094-organizer-extract-benchmark-v1",
        "protocol": {
            "latency_b1": f"median full extract cycle, batch=1, {a.latency_runs} runs after {a.latency_warmup} warmups",
            "throughput": f"batches 1/8/16/32, each >= {a.throughput_seconds:.1f}s; best FPS counts",
            "included": "disk read, decode, bbox crop, preprocessing, both ONNX forwards, postprocessing, L2/ensemble embedding",
            "excluded": "gallery search and re-ranking",
        },
        "members": [{"name": x["name"], "weight": x["weight"], "provider": x["provider"]} for x in sessions],
        "objects": int(len(combined)),
        "workers": int(a.workers),
        "model_load_s_reference": float(model_load_s),
        "onnx_weight_bytes": int(onnx_weight_bytes),
        "onnx_weight_gib": float(onnx_weight_bytes / 1024**3),
        "deterministic_repeat": deterministic,
        "determinism_max_abs": determinism_max_abs,
        "latency_b1_ms_median": float(statistics.median(latency_ms)),
        "latency_b1_ms_p95": float(np.percentile(latency_ms, 95)),
        "latency_full_score": bool(statistics.median(latency_ms) <= 40.0),
        "throughput": throughput,
        "best_throughput_fps": float(best["fps"]),
        "best_batch": int(best["batch"]),
        "throughput_full_score": bool(best["fps"] >= 100.0),
    }
    (out / "official_extract_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(throughput).to_csv(out / "throughput.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
