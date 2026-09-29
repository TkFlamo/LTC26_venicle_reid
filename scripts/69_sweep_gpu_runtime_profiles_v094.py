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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.data.prepare import prepare_hackathon_dataset
from vehicle_fingerprint.official_eval import blended_embedding
from vehicle_fingerprint.runtime_pt import (
    _make_dataset_loader,
    configure_loaded_models_runtime,
    forward_loaded_models,
    load_multi_pt_models,
    prepare_input_tensor,
)
from vehicle_fingerprint.score_ensemble import read_score_ensemble_deployment, weighted_concat_embeddings

EXPECTED = {"base_full": 0.65, "external_convnext_full": 0.35}


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def resolve(root: Path, value, fallback=None) -> Path:
    p = Path(value or fallback).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def build_specs(dep: Path) -> list[dict]:
    meta = read_score_ensemble_deployment(dep)
    got = {m["name"]: float(m["weight"]) for m in meta["members"]}
    if set(got) != set(EXPECTED) or any(abs(got[k] - v) > 1e-9 for k, v in EXPECTED.items()):
        raise RuntimeError(f"Expected {EXPECTED}, got {got}")
    specs = []
    for member in meta["members"]:
        mdir = Path(member["path"])
        dm = json.loads((mdir / "deployment.json").read_text(encoding="utf-8"))
        ck = resolve(mdir, dm.get("checkpoint"), "reid.pt")
        rr = json.loads(resolve(mdir, dm.get("retrieval_recipe"), "retrieval_recipe.json").read_text(encoding="utf-8"))
        specs.append({
            "name": member["name"],
            "weight": float(member["weight"]),
            "checkpoint": ck,
            "base_alpha": float(rr.get("base_alpha", 0.0)),
        })
    return specs


def materialize_embedding(outs, models) -> np.ndarray:
    members, weights = [], []
    for info, out in zip(models, outs):
        zg = out["z_global"].float().cpu().numpy()
        zf = out["z_fused"].float().cpu().numpy()
        members.append(blended_embedding(zg, zf, float(info.get("base_alpha", 0.0))))
        weights.append(float(info["weight"]))
    return weighted_concat_embeddings(members, weights)


def one_embedding(batch, models, dev, precision, *, channels_last, native_fp16, parallel_mode, microbatch):
    x = prepare_input_tensor(
        batch, dev, precision=precision, channels_last=channels_last, native_fp16=native_fp16
    )
    outs = forward_loaded_models(
        models, x, dev, precision=precision, parallel_mode=parallel_mode, stream_microbatch=microbatch
    )
    emb = materialize_embedding(outs, models)
    sync(dev)
    return emb


def bench(batch, models, dev, precision, seconds, *, channels_last, native_fp16, parallel_mode, microbatch):
    x = prepare_input_tensor(
        batch, dev, precision=precision, channels_last=channels_last, native_fp16=native_fp16
    )
    for _ in range(6):
        _ = forward_loaded_models(
            models, x, dev, precision=precision, parallel_mode=parallel_mode, stream_microbatch=microbatch
        )
    sync(dev)
    n = 0
    t0 = time.perf_counter()
    while True:
        _ = forward_loaded_models(
            models, x, dev, precision=precision, parallel_mode=parallel_mode, stream_microbatch=microbatch
        )
        n += int(x.shape[0])
        if time.perf_counter() - t0 >= seconds:
            sync(dev)
            dt = time.perf_counter() - t0
            break
    return {"samples": n, "seconds": dt, "fps": n / dt}


def similarity(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float32); b = b.astype(np.float32)
    max_abs = float(np.max(np.abs(a - b)))
    an = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-12)
    cos = np.sum(an * bn, axis=1)
    return {"max_abs": max_abs, "mean_cosine": float(np.mean(cos)), "min_cosine": float(np.min(cos))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep CUDA runtime profiles for the fixed v0.9.4 dual-PT ensemble")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--deployment-dir", default="deploy/models_current")
    ap.add_argument("--out", default="artifacts/gpu_runtime_profile_sweep")
    ap.add_argument("--device", default="0")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--include-native-fp16", action="store_true")
    a = ap.parse_args()

    inp = Path(a.input_dir).expanduser().resolve()
    dep = Path(a.deployment_dir).expanduser()
    if not dep.is_absolute(): dep = (ROOT / dep).resolve()
    out = Path(a.out).expanduser()
    if not out.is_absolute(): out = (ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    specs = build_specs(dep)

    q = pd.read_csv(inp / "test_query.csv", dtype={"image_id": str})
    g = pd.read_csv(inp / "test_gallery.csv", dtype={"image_id": str})
    combined = pd.concat([q, g], ignore_index=True)

    with tempfile.TemporaryDirectory(prefix="v094_gpu_sweep_") as td:
        td = Path(td)
        csv = td / "objects.csv"; combined.to_csv(csv, index=False)
        prepdir = td / "prepared"
        prepare_hackathon_dataset(csv, inp / "images", prepdir, pad=.03, val_fraction=.0, eval_fraction=.0, materialize_crops=False)
        manifest = prepdir / ("test.csv" if (prepdir / "test.csv").is_file() else "manifest.csv")

        # Load once in canonical FP32-parameter form.  Benchmark NCHW first, then convert layout to CL.
        models, dev, size, init = load_multi_pt_models(specs, device=a.device)
        _, loader = _make_dataset_loader(manifest, size, init["preprocess"], batch_size=a.batch, workers=0)
        batch = next(iter(loader))

        rows = []
        configure_loaded_models_runtime(models, dev, precision=a.precision, channels_last=False, native_fp16=False)
        baseline = one_embedding(batch, models, dev, a.precision, channels_last=False, native_fp16=False,
                                 parallel_mode="sequential", microbatch=0)

        profiles = [
            ("nchw_sequential", False, False, "sequential", 0),
            ("nchw_streams", False, False, "streams", 0),
        ]
        for name, cl, half, mode, mb in profiles:
            r = bench(batch, models, dev, a.precision, a.seconds, channels_last=cl, native_fp16=half,
                      parallel_mode=mode, microbatch=mb)
            e = one_embedding(batch, models, dev, a.precision, channels_last=cl, native_fp16=half,
                              parallel_mode=mode, microbatch=mb)
            rows.append({"profile": name, "channels_last": cl, "native_fp16": half,
                         "parallel_mode": mode, "stream_microbatch": mb, **r, **similarity(baseline, e)})

        configure_loaded_models_runtime(models, dev, precision=a.precision, channels_last=True, native_fp16=False)
        candidates = [("channels_last_sequential", "sequential", 0), ("channels_last_streams", "streams", 0)]
        for mb in sorted({x for x in (4, 8, 16) if x < a.batch}):
            candidates.append((f"channels_last_streams_mb{mb}", "streams", mb))
        for name, mode, mb in candidates:
            r = bench(batch, models, dev, a.precision, a.seconds, channels_last=True, native_fp16=False,
                      parallel_mode=mode, microbatch=mb)
            e = one_embedding(batch, models, dev, a.precision, channels_last=True, native_fp16=False,
                              parallel_mode=mode, microbatch=mb)
            rows.append({"profile": name, "channels_last": True, "native_fp16": False,
                         "parallel_mode": mode, "stream_microbatch": mb, **r, **similarity(baseline, e)})

        if a.include_native_fp16 and a.precision == "fp16":
            # Reload so the FP16 conversion cannot affect the canonical profiles above.
            half_models, half_dev, half_size, half_init = load_multi_pt_models(specs, device=a.device)
            if tuple(half_size) != tuple(size): raise RuntimeError("Reloaded model preprocessing changed unexpectedly")
            configure_loaded_models_runtime(half_models, half_dev, precision="fp16", channels_last=True, native_fp16=True)
            half_candidates = [("native_fp16_cl_streams", "streams", 0)]
            for mb in sorted({x for x in (4, 8, 16) if x < a.batch}):
                half_candidates.append((f"native_fp16_cl_streams_mb{mb}", "streams", mb))
            for name, mode, mb in half_candidates:
                r = bench(batch, half_models, half_dev, "fp16", a.seconds, channels_last=True, native_fp16=True,
                          parallel_mode=mode, microbatch=mb)
                e = one_embedding(batch, half_models, half_dev, "fp16", channels_last=True, native_fp16=True,
                                  parallel_mode=mode, microbatch=mb)
                rows.append({"profile": name, "channels_last": True, "native_fp16": True,
                             "parallel_mode": mode, "stream_microbatch": mb, **r, **similarity(baseline, e)})

    valid = [r for r in rows if r["mean_cosine"] >= 0.9999 and r["max_abs"] <= 0.01]
    best = max(valid or rows, key=lambda r: r["fps"])
    args = ["--parallel-mode", best["parallel_mode"], "--stream-microbatch", str(best["stream_microbatch"])]
    if not best["channels_last"]: args.append("--no-channels-last")
    if best["native_fp16"]: args.append("--native-fp16")
    report = {
        "schema": "vehicle-reid-v094-gpu-runtime-profile-sweep-v1",
        "device": str(dev),
        "precision": a.precision,
        "batch": int(a.batch),
        "seconds_per_profile": float(a.seconds),
        "results": rows,
        "recommended": best,
        "recommended_args": args,
        "note": "Sweep is GPU-forward-only on one real preprocessed batch; run script 63 with recommended_args for organizer-style end-to-end FPS.",
    }
    (out / "gpu_runtime_profiles.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rows).sort_values("fps", ascending=False).to_csv(out / "gpu_runtime_profiles.csv", index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("[RECOMMENDED]", " ".join(args))
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
