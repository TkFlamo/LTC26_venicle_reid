#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
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
from vehicle_fingerprint.score_ensemble import read_score_ensemble_deployment, weighted_concat_embeddings
from vehicle_fingerprint.utils import autocast_context

EXPECTED = {"base_full": 0.65, "external_convnext_full": 0.35}


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def resolve(root: Path, value, fallback=None) -> Path:
    value = value or fallback
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def build_specs(meta: dict) -> list[dict]:
    got = {m["name"]: float(m["weight"]) for m in meta["members"]}
    if set(got) != set(EXPECTED) or any(abs(got[k] - v) > 1e-9 for k, v in EXPECTED.items()):
        raise RuntimeError(f"Expected fixed production ensemble {EXPECTED}; got {got}")
    specs = []
    for member in meta["members"]:
        dep = Path(member["path"])
        dm = json.loads((dep / "deployment.json").read_text(encoding="utf-8"))
        checkpoint = resolve(dep, dm.get("checkpoint"), "reid.pt")
        recipe = json.loads(resolve(dep, dm.get("retrieval_recipe"), "retrieval_recipe.json").read_text(encoding="utf-8"))
        specs.append({
            "name": member["name"],
            "weight": float(member["weight"]),
            "checkpoint": checkpoint,
            "base_alpha": float(recipe.get("base_alpha", 0.0)),
        })
    return specs


def next_cycling(loader, state):
    try:
        return next(state[0])
    except StopIteration:
        state[0] = iter(loader)
        return next(state[0])


def full_extract(batch, models, dev, precision):
    t0 = time.perf_counter()
    x = batch["image"].to(dev, non_blocking=True)
    t1 = time.perf_counter()
    with torch.inference_mode(), autocast_context(dev, precision):
        outs = [info["model"](x) for info in models]
    t2 = time.perf_counter()

    member_emb, weights = [], []
    for info, out in zip(models, outs):
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
    emb = weighted_concat_embeddings(member_emb, weights)
    t3 = time.perf_counter()
    return emb, (t1 - t0), (t2 - t1), (t3 - t2)


def bench_loader(loader, seconds: float) -> dict:
    state = [iter(loader)]
    # Prime worker processes and prefetch queue outside the measurement.
    for _ in range(3):
        _ = next_cycling(loader, state)
    n = 0
    start = time.perf_counter()
    while True:
        b = next_cycling(loader, state)
        n += int(b["image"].shape[0])
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            break
    return {"samples": n, "seconds": elapsed, "fps": n / elapsed}


def bench_forward_static(x: torch.Tensor, models, dev, precision, seconds: float) -> dict:
    # Combined two-backbone GPU-only throughput. No H2D, D2H or CPU postprocess.
    with torch.inference_mode(), autocast_context(dev, precision):
        for _ in range(5):
            for info in models:
                _ = info["model"](x)
    sync(dev)
    n = 0
    start = time.perf_counter()
    while True:
        with torch.inference_mode(), autocast_context(dev, precision):
            for info in models:
                _ = info["model"](x)
        n += int(x.shape[0])
        if time.perf_counter() - start >= seconds:
            sync(dev)
            elapsed = time.perf_counter() - start
            break
    return {"samples": n, "seconds": elapsed, "fps": n / elapsed}


def bench_member_static(x: torch.Tensor, info, dev, precision, seconds: float) -> dict:
    with torch.inference_mode(), autocast_context(dev, precision):
        for _ in range(5):
            _ = info["model"](x)
    sync(dev)
    n = 0
    start = time.perf_counter()
    while True:
        with torch.inference_mode(), autocast_context(dev, precision):
            _ = info["model"](x)
        n += int(x.shape[0])
        if time.perf_counter() - start >= seconds:
            sync(dev)
            elapsed = time.perf_counter() - start
            break
    return {"samples": n, "seconds": elapsed, "fps": n / elapsed}


def bench_full(loader, models, dev, precision, seconds: float) -> dict:
    state = [iter(loader)]
    for _ in range(3):
        b = next_cycling(loader, state)
        _ = full_extract(b, models, dev, precision)
    sync(dev)
    n = 0
    wait = h2d = fwd = post = 0.0
    start = time.perf_counter()
    while True:
        tw = time.perf_counter()
        b = next_cycling(loader, state)
        wait += time.perf_counter() - tw
        emb, a, bb, c = full_extract(b, models, dev, precision)
        h2d += a; fwd += bb; post += c
        n += int(emb.shape[0])
        if time.perf_counter() - start >= seconds:
            sync(dev)
            elapsed = time.perf_counter() - start
            break
    return {
        "samples": n, "seconds": elapsed, "fps": n / elapsed,
        "loader_wait_ms_per_sample": 1000 * wait / max(n, 1),
        "h2d_submit_ms_per_sample": 1000 * h2d / max(n, 1),
        # fwd is CPU submit time unless a D2H later forces completion; elapsed/fps is authoritative.
        "forward_submit_ms_per_sample": 1000 * fwd / max(n, 1),
        "postprocess_sync_ms_per_sample": 1000 * post / max(n, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose PT extract bottlenecks for the fixed 0.65/0.35 ensemble")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--deployment-dir", default="deploy/models_current")
    ap.add_argument("--out", default="artifacts/extract_diagnostics_pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--workers", default="0,2,4,8,12,16")
    ap.add_argument("--batches", default="1,8,16,32")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    a = ap.parse_args()

    workers = sorted({max(0, int(x)) for x in a.workers.split(",") if x.strip()})
    batches = sorted({max(1, int(x)) for x in a.batches.split(",") if x.strip()})
    inp = Path(a.input_dir).expanduser().resolve()
    dep = Path(a.deployment_dir).expanduser()
    if not dep.is_absolute(): dep = (ROOT / dep).resolve()
    out = Path(a.out).expanduser()
    if not out.is_absolute(): out = (ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    meta = read_score_ensemble_deployment(dep)
    specs = build_specs(meta)
    models, dev, size, init = load_multi_pt_models(specs, device=a.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    q = pd.read_csv(inp / "test_query.csv", dtype={"image_id": str})
    g = pd.read_csv(inp / "test_gallery.csv", dtype={"image_id": str})
    combined = pd.concat([q, g], ignore_index=True)

    with tempfile.TemporaryDirectory(prefix="v094_diag_") as td:
        td = Path(td)
        csv_path = td / "objects.csv"; combined.to_csv(csv_path, index=False)
        prep_dir = td / "prepared"
        prepare_hackathon_dataset(csv_path, inp / "images", prep_dir, pad=.03, val_fraction=.0, eval_fraction=.0, materialize_crops=False)
        manifest = prep_dir / ("test.csv" if (prep_dir / "test.csv").is_file() else "manifest.csv")
        prep = init["preprocess"]

        rows = []
        for bsz in batches:
            # One real batch for GPU-only profiling.
            _, static_loader = _make_dataset_loader(manifest, size, prep, batch_size=bsz, workers=0, prefetch_factor=a.prefetch_factor)
            static_batch = next(iter(static_loader))
            x = static_batch["image"].to(dev, non_blocking=False)
            sync(dev)
            comb = bench_forward_static(x, models, dev, a.precision, a.seconds)
            rows.append({"kind":"gpu_forward_only", "batch":bsz, "workers":None, **comb})
            for info in models:
                one = bench_member_static(x, info, dev, a.precision, max(2.0, a.seconds / 2))
                rows.append({"kind":f"gpu_forward_{info['name']}", "batch":bsz, "workers":None, **one})
            del x
            if dev.type == "cuda": torch.cuda.empty_cache()

            for w in workers:
                _, loader = _make_dataset_loader(manifest, size, prep, batch_size=bsz, workers=w, prefetch_factor=a.prefetch_factor)
                data = bench_loader(loader, a.seconds)
                rows.append({"kind":"input_pipeline_only", "batch":bsz, "workers":w, **data})
                # Recreate loader so the full test starts with a fresh, primed queue.
                _, loader = _make_dataset_loader(manifest, size, prep, batch_size=bsz, workers=w, prefetch_factor=a.prefetch_factor)
                full = bench_full(loader, models, dev, a.precision, a.seconds)
                rows.append({"kind":"full_extract", "batch":bsz, "workers":w, **full})

    props = torch.cuda.get_device_properties(dev) if dev.type == "cuda" else None
    environment = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cpu_count_logical": os.cpu_count(),
        "gpu": None if props is None else {
            "name": props.name,
            "total_memory_gib": props.total_memory / 1024**3,
            "compute_capability": [props.major, props.minor],
            "multi_processor_count": props.multi_processor_count,
        },
    }
    report = {
        "schema": "vehicle-reid-v094-pt-extract-diagnostics-v1",
        "environment": environment,
        "objects": len(combined),
        "image_size": list(map(int, size)),
        "precision": a.precision,
        "prefetch_factor": a.prefetch_factor,
        "results": rows,
    }
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "diagnostics.csv", index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
