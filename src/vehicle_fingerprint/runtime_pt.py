from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.dataset import ReIDDataset
from .features import META_KEYS, load_inference_model
from .utils import autocast_context

FULL_OUTPUTS = ("z_fused", "z_global", "z_local", "parts", "visibility", "visibility_score", "local")


def _now() -> float:
    return time.perf_counter()


def _sync_cuda(device: torch.device | str | None) -> None:
    if device is None:
        return
    try:
        d = torch.device(device)
    except Exception:
        return
    if d.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(d)


def _string_array(values):
    return np.asarray([str(x) for x in values], dtype=np.str_)


def _collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = [x[k] for x in batch] if k in META_KEYS else torch.stack([x[k] for x in batch])
    return out


def _worker_init(worker_id: int) -> None:
    """Keep each loader worker single-threaded.

    OpenCV/BLAS can otherwise create their own thread pools inside every
    DataLoader subprocess.  That is especially painful on Windows spawn and on
    the 128-logical-CPU organizer host where oversubscription can dominate the
    actual JPEG/resize/color work.
    """
    del worker_id
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _make_dataset_loader(manifest, image_size, prep, batch_size, workers, *, prefetch_factor=2):
    df = pd.read_csv(manifest)
    tmp = df.copy()
    if "vehicle_id" not in tmp.columns:
        tmp["vehicle_id"] = "unknown"
    if "dataset" not in tmp.columns:
        tmp["dataset"] = "input"
    label_map = {
        k: i for i, k in enumerate(sorted(set(tmp.dataset.astype(str) + ":" + tmp.vehicle_id.astype(str))))
    }
    ds = ReIDDataset(
        tmp,
        image_size=image_size,
        train=False,
        label_map=label_map,
        augmentation_profile=prep.get("augmentation_profile", "baseline_v4"),
        use_source_bbox=prep.get("use_source_bbox", True),
        bbox_pad=prep.get("bbox_pad", .03),
        inference_only=True,
        return_color_descriptor=True,
    )
    kwargs = dict(
        batch_size=int(batch_size), shuffle=False, num_workers=int(workers),
        pin_memory=torch.cuda.is_available(), collate_fn=_collate,
    )
    if int(workers) > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=max(1, int(prefetch_factor)),
            worker_init_fn=_worker_init,
        )
    return df, DataLoader(ds, **kwargs)


def _new_acc() -> dict[str, list]:
    return {k: [] for k in [
        "z_fused", "z_global", "z_local", "parts", "visibility", "visibility_score", "local", "color",
        "sample_id", "vehicle_key", "camera_id", "path",
    ]}


def _append_outputs(acc: dict[str, list], out: dict[str, torch.Tensor]) -> None:
    acc["z_fused"].append(out["z_fused"].float().cpu().numpy())
    acc["z_global"].append(out["z_global"].float().cpu().numpy())
    acc["z_local"].append(out["z_local"].float().cpu().numpy())
    acc["parts"].append(out["parts"].float().cpu().numpy().astype(np.float16))
    acc["visibility"].append(out["visibility"].cpu().numpy().astype(np.uint8))
    acc["visibility_score"].append(out["visibility_score"].float().cpu().numpy().astype(np.float16))
    acc["local"].append(out["local"].float().cpu().numpy().astype(np.float16))


def _append_shared_meta(acc: dict[str, list], batch: dict[str, Any], color_np: np.ndarray) -> None:
    acc["color"].extend(color_np)
    acc["sample_id"].extend(batch["sample_id"])
    acc["vehicle_key"].extend(batch["vehicle_key"])
    acc["camera_id"].extend(batch["camera_id"])
    acc["path"].extend(batch["path"])


def _finalize_cache(df: pd.DataFrame, acc: dict[str, list]) -> dict[str, np.ndarray]:
    payload = {
        "z_fused": np.concatenate(acc["z_fused"]).astype(np.float32),
        "z_global": np.concatenate(acc["z_global"]).astype(np.float32),
        "z_local": np.concatenate(acc["z_local"]).astype(np.float32),
        "parts": np.concatenate(acc["parts"]).astype(np.float16),
        "visibility": np.concatenate(acc["visibility"]).astype(np.uint8),
        "visibility_score": np.concatenate(acc["visibility_score"]).astype(np.float16),
        "local": np.concatenate(acc["local"]).astype(np.float16),
        "color": np.stack(acc["color"]).astype(np.float32),
        "sample_id": _string_array(acc["sample_id"]),
        "vehicle_key": _string_array(acc["vehicle_key"]),
        "camera_id": _string_array(acc["camera_id"]),
        "path": _string_array(acc["path"]),
    }
    for col in ("image_id", "source_row", "dataset", "split"):
        if col in df.columns:
            payload[f"meta_{col}"] = _string_array(df[col].astype(str).tolist())
    return payload


def _prep_signature(mcfg: dict) -> tuple:
    prep = dict(mcfg.get("preprocess", {}) or {})
    size = tuple(map(int, prep.get("image_size", [256, 384])))
    return (
        size,
        prep.get("augmentation_profile", "baseline_v4"),
        bool(prep.get("use_source_bbox", True)),
        float(prep.get("bbox_pad", .03)),
    )


def load_multi_pt_models(specs: list[dict], *, device="0") -> tuple[list[dict], torch.device, tuple, dict]:
    """Load several PT feature models while enforcing one shared preprocessing contract."""
    if not specs:
        raise ValueError("No PT model specs supplied")

    loaded = []
    signatures = []
    dev0 = None
    t0 = _now()
    for spec in specs:
        model, dev, mcfg = load_inference_model(spec["checkpoint"], device)
        if dev0 is None:
            dev0 = dev
        elif torch.device(dev) != torch.device(dev0):
            raise RuntimeError(f"All ensemble models must use the same device: {dev0} vs {dev}")
        sig = _prep_signature(mcfg)
        signatures.append(sig)
        loaded.append({**spec, "model": model, "model_cfg": mcfg})
    _sync_cuda(dev0)
    if len(set(signatures)) != 1:
        raise RuntimeError(
            "Production ensemble members have different preprocessing; shared PT inference would change the model contract: "
            f"{signatures}"
        )
    size, augmentation_profile, use_source_bbox, bbox_pad = signatures[0]
    prep = {
        "image_size": list(size),
        "augmentation_profile": augmentation_profile,
        "use_source_bbox": use_source_bbox,
        "bbox_pad": bbox_pad,
    }
    return loaded, torch.device(dev0), size, {"load_s": _now() - t0, "preprocess": prep}


def extract_feature_caches_loaded_pt_timed(
    manifest,
    models: list[dict],
    dev: torch.device,
    size: tuple,
    prep: dict,
    *,
    precision="fp16",
    batch_size=32,
    workers=8,
    warmup_batches=2,
):
    """Extract several member caches with already-loaded PT models."""
    t_all = _now()
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    t = _now()
    df, loader = _make_dataset_loader(manifest, size, prep, batch_size, workers)
    loader_init_s = _now() - t

    accs = {str(x["name"]): _new_acc() for x in models}
    member_forward = {str(x["name"]): 0.0 for x in models}
    stats = {
        "runtime": "pytorch_multi",
        "samples": int(len(df)),
        "batch_size": int(batch_size),
        "workers": int(workers),
        "precision": str(precision),
        "device": str(dev),
        "image_size": list(map(int, size)),
        "dataset_loader_init_s": float(loader_init_s),
        "input_wait_preprocess_s": 0.0,
        "input_conversion_or_h2d_s": 0.0,
        "neural_forward_s": 0.0,
        "feature_postprocess_s": 0.0,
        "shared_preprocess": True,
        "members": {},
    }

    it = iter(loader)
    bi = 0
    while True:
        t = _now()
        try:
            batch = next(it)
        except StopIteration:
            break
        stats["input_wait_preprocess_s"] += _now() - t

        t = _now()
        x = batch["image"].to(dev, non_blocking=True)
        _sync_cuda(dev)
        stats["input_conversion_or_h2d_s"] += _now() - t

        if bi < int(warmup_batches):
            with torch.inference_mode(), autocast_context(dev, precision):
                for info in models:
                    _ = info["model"](x)
            _sync_cuda(dev)

        outs: dict[str, dict[str, torch.Tensor]] = {}
        for info in models:
            name = str(info["name"])
            t = _now()
            with torch.inference_mode(), autocast_context(dev, precision):
                outs[name] = info["model"](x)
            _sync_cuda(dev)
            dt = _now() - t
            member_forward[name] += dt
            stats["neural_forward_s"] += dt

        t = _now()
        color_np = batch["color"].float().cpu().numpy()
        for info in models:
            name = str(info["name"])
            _append_outputs(accs[name], outs[name])
            _append_shared_meta(accs[name], batch, color_np)
        stats["feature_postprocess_s"] += _now() - t
        bi += 1

    payloads = {name: _finalize_cache(df, acc) for name, acc in accs.items()}
    stats["total_s"] = _now() - t_all
    stats["throughput_samples_s"] = float(len(df) / max(stats["total_s"], 1e-9))
    stats["neural_ms_per_sample"] = float(1000.0 * stats["neural_forward_s"] / max(len(df), 1))
    for info in models:
        name = str(info["name"])
        stats["members"][name] = {
            "checkpoint": str(Path(info["checkpoint"]).resolve()),
            "neural_forward_s": float(member_forward[name]),
            "neural_ms_per_sample": float(1000.0 * member_forward[name] / max(len(df), 1)),
        }
    return payloads, stats


def extract_feature_caches_multi_pt_timed(
    manifest,
    specs: list[dict],
    *,
    device="0",
    precision="fp16",
    batch_size=32,
    workers=8,
    warmup_batches=2,
):
    """Convenience wrapper that loads the models and extracts one manifest."""
    models, dev, size, init = load_multi_pt_models(specs, device=device)
    payloads, stats = extract_feature_caches_loaded_pt_timed(
        manifest, models, dev, size, init["preprocess"],
        precision=precision, batch_size=batch_size, workers=workers, warmup_batches=warmup_batches,
    )
    stats["model_load_s"] = float(init["load_s"])
    stats["total_s_including_model_load"] = float(stats["total_s"] + init["load_s"])
    return payloads, stats
