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
from vehicle_fingerprint.runtime_pt import _make_dataset_loader, load_multi_pt_models
from vehicle_fingerprint.score_ensemble import weighted_concat_embeddings
from vehicle_fingerprint.utils import autocast_context

WEIGHT_EXTS = {".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".npz"}


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)


def _resolve(root: Path, value, fallback=None) -> Path:
    value = value or fallback
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def _member_dir(dep: Path, member: dict) -> Path:
    p = Path(str(member.get("path", ""))).expanduser()
    if not p.is_absolute():
        p = dep / p
    return p.resolve()


def _member_spec(dep: Path, member: dict, *, runtime_weight: float) -> dict:
    mdir = _member_dir(dep, member)
    mp = mdir / "deployment.json"
    dm = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
    checkpoint = _resolve(mdir, dm.get("checkpoint"), "reid.pt")
    recipe_path = _resolve(mdir, dm.get("retrieval_recipe"), "retrieval_recipe.json")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{member.get('name')}: checkpoint missing: {checkpoint}")
    if not recipe_path.is_file():
        raise FileNotFoundError(f"{member.get('name')}: retrieval recipe missing: {recipe_path}")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    return {
        "name": str(member["name"]),
        "weight": float(runtime_weight),
        "checkpoint": checkpoint,
        "base_alpha": float(recipe.get("base_alpha", 0.0)),
    }


def _build_specs(dep: Path, meta: dict, runtime_member: str) -> list[dict]:
    members = {str(m.get("name")): m for m in meta.get("members", [])}
    if runtime_member == "ensemble":
        if not members:
            raise RuntimeError("Top-level deployment has no score-ensemble members")
        raw = [float(m.get("weight", 1.0)) for m in meta["members"]]
        total = sum(raw)
        if total <= 0:
            raise RuntimeError("Ensemble weights must sum to a positive value")
        return [
            _member_spec(dep, m, runtime_weight=w / total)
            for m, w in zip(meta["members"], raw)
        ]
    if runtime_member not in members:
        raise RuntimeError(f"Requested member {runtime_member!r} is missing; available={sorted(members)}")
    return [_member_spec(dep, members[runtime_member], runtime_weight=1.0)]


def _batch_extract(batch: dict, models: list[dict], dev: torch.device, precision: str) -> np.ndarray:
    x = batch["image"].to(dev, non_blocking=True)
    member_emb = []
    weights = []

    # Deferred materialization: all selected neural forwards happen before D2H.
    with torch.inference_mode(), autocast_context(dev, precision):
        outs = [info["model"](x) for info in models]

    for info, out in zip(models, outs):
        # Materialize the same feature families used by the single/ensemble retrieval stack.
        zg = out["z_global"].float().cpu().numpy()
        zf = out["z_fused"].float().cpu().numpy()
        _ = out["z_local"].float().cpu().numpy()
        _ = out["parts"].float().cpu().numpy().astype(np.float16)
        _ = out["visibility"].cpu().numpy().astype(np.uint8)
        _ = out["visibility_score"].float().cpu().numpy().astype(np.float16)
        _ = out["local"].float().cpu().numpy().astype(np.float16)
        member_emb.append(blended_embedding(zg, zf, float(info["base_alpha"])))
        weights.append(float(info["weight"]))

    if "color" in batch:
        _ = batch["color"].float().cpu().numpy()

    # Single-member production must return the native base_full embedding, not a
    # synthetic concatenation.  Ensemble compatibility is kept behind --member ensemble.
    if len(member_emb) == 1:
        return member_emb[0].astype(np.float32, copy=False)
    return weighted_concat_embeddings(member_emb, weights)


def _next_cycling(loader, state):
    try:
        return next(state[0])
    except StopIteration:
        state[0] = iter(loader)
        return next(state[0])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Organizer-style PT extract benchmark. Default runtime benchmarks base_full only; "
                    "external ConvNeXt remains packaged but disabled."
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--deployment-dir", default="deploy/models_current")
    ap.add_argument("--out", default="artifacts/official_extract_benchmark_pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=2, help="Batches prefetched per DataLoader worker")
    ap.add_argument(
        "--member", choices=["base_full", "external_convnext_full", "ensemble"], default="base_full",
        help="Runtime model to benchmark. Default base_full. 'ensemble' restores the old two-model benchmark.",
    )
    ap.add_argument("--latency-warmup", type=int, default=50)
    ap.add_argument("--latency-runs", type=int, default=300)
    ap.add_argument("--throughput-seconds", type=float, default=10.0)
    a = ap.parse_args()

    if int(a.workers) == 0:
        print(
            "[WARN] --workers=0 makes JPEG decode, crop, resize and color descriptor run serially. "
            "Use --workers 8 first and sweep the target host if necessary.",
            flush=True,
        )

    inp = Path(a.input_dir).expanduser().resolve()
    dep = Path(a.deployment_dir).expanduser()
    if not dep.is_absolute():
        dep = (ROOT / dep).resolve()
    out = Path(a.out).expanduser()
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    meta_path = dep / "deployment.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("mode", "")).lower() != "score_ensemble":
        raise RuntimeError(f"Expected top-level score_ensemble deployment, got {meta.get('mode')!r}")
    specs = _build_specs(dep, meta, a.member)

    t_load = time.perf_counter()
    models, dev, size, init = load_multi_pt_models(specs, device=a.device)
    model_load_s = time.perf_counter() - t_load
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Report both active runtime bytes and total packaged bytes.  ConvNeXt stays
    # in the package by design even when it is not loaded for base_full-only inference.
    active_weight_files = [Path(x["checkpoint"]) for x in specs]
    active_weight_bytes = sum(p.stat().st_size for p in active_weight_files if p.is_file())
    package_weight_files = [p for p in dep.rglob("*") if p.is_file() and p.suffix.lower() in WEIGHT_EXTS]
    package_weight_bytes = sum(p.stat().st_size for p in package_weight_files)

    q = pd.read_csv(inp / "test_query.csv", dtype={"image_id": str})
    g = pd.read_csv(inp / "test_gallery.csv", dtype={"image_id": str})
    combined = pd.concat([q, g], ignore_index=True)

    with tempfile.TemporaryDirectory(prefix="v094_official_pt_bench_") as td:
        td = Path(td)
        csv_path = td / "objects.csv"
        combined.to_csv(csv_path, index=False)
        prep_dir = td / "prepared"
        prepare_hackathon_dataset(
            csv_path, inp / "images", prep_dir, pad=.03, val_fraction=.0,
            eval_fraction=.0, materialize_crops=False,
        )
        manifest = prep_dir / ("test.csv" if (prep_dir / "test.csv").is_file() else "manifest.csv")
        prep = init["preprocess"]

        _, det_loader = _make_dataset_loader(
            manifest, size, prep, batch_size=8, workers=0, prefetch_factor=a.prefetch_factor
        )
        det_batch = next(iter(det_loader))
        _sync(dev); det_a = _batch_extract(det_batch, models, dev, a.precision); _sync(dev)
        det_b = _batch_extract(det_batch, models, dev, a.precision); _sync(dev)
        determinism_max_abs = float(np.max(np.abs(det_a - det_b)))
        deterministic = bool(np.array_equal(det_a, det_b) or np.allclose(det_a, det_b, rtol=0.0, atol=1e-6))

        # Organizer latency protocol intentionally includes serial input work for batch=1.
        _, latency_loader = _make_dataset_loader(
            manifest, size, prep, batch_size=1, workers=0, prefetch_factor=a.prefetch_factor
        )
        state = [iter(latency_loader)]
        for _ in range(int(a.latency_warmup)):
            b = _next_cycling(latency_loader, state)
            _sync(dev); _batch_extract(b, models, dev, a.precision); _sync(dev)
        latency_ms = []
        for _ in range(int(a.latency_runs)):
            _sync(dev)
            t = time.perf_counter()
            b = _next_cycling(latency_loader, state)
            _batch_extract(b, models, dev, a.precision)
            _sync(dev)
            latency_ms.append((time.perf_counter() - t) * 1000.0)

        throughput = []
        for batch_size in (1, 8, 16, 32):
            _, loader = _make_dataset_loader(
                manifest, size, prep, batch_size=batch_size, workers=a.workers,
                prefetch_factor=a.prefetch_factor,
            )
            state = [iter(loader)]
            for _ in range(3):
                b = _next_cycling(loader, state)
                _batch_extract(b, models, dev, a.precision)
            _sync(dev)
            start = time.perf_counter(); n = 0
            while True:
                b = _next_cycling(loader, state)
                n += int(_batch_extract(b, models, dev, a.precision).shape[0])
                if time.perf_counter() - start >= float(a.throughput_seconds):
                    _sync(dev)
                    elapsed = time.perf_counter() - start
                    break
            throughput.append({"batch": batch_size, "samples": n, "seconds": elapsed, "fps": n / elapsed})

    best = max(throughput, key=lambda x: x["fps"])
    forward_count = len(specs)
    report = {
        "schema": "vehicle-reid-v094-organizer-extract-pt-benchmark-v2",
        "runtime_profile": a.member,
        "protocol": {
            "latency_b1": f"median full extract cycle, batch=1, {a.latency_runs} runs after {a.latency_warmup} warmups",
            "throughput": f"batches 1/8/16/32, each >= {a.throughput_seconds:.1f}s; best FPS counts",
            "included": f"disk read, decode, bbox crop, preprocessing, {forward_count} PyTorch forward(s), feature materialization, L2 embedding",
            "excluded": "gallery search and re-ranking",
        },
        "members": [
            {"name": x["name"], "weight": x["weight"], "checkpoint": str(x["checkpoint"])} for x in specs
        ],
        "packaged_external_convnext_enabled": bool(a.member in {"external_convnext_full", "ensemble"}),
        "objects": int(len(combined)),
        "workers": int(a.workers),
        "prefetch_factor": int(a.prefetch_factor),
        "precision": a.precision,
        "device": str(dev),
        "model_load_s_reference": float(model_load_s),
        "active_pt_weight_bytes": int(active_weight_bytes),
        "active_pt_weight_gib": float(active_weight_bytes / 1024**3),
        "package_pt_weight_bytes": int(package_weight_bytes),
        "package_pt_weight_gib": float(package_weight_bytes / 1024**3),
        # Compatibility aliases: these now mean active runtime weights.
        "pt_weight_bytes": int(active_weight_bytes),
        "pt_weight_gib": float(active_weight_bytes / 1024**3),
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
    (out / "official_extract_benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(throughput).to_csv(out / "throughput.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
