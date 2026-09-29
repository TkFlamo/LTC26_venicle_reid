from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data.color import robust_color_descriptor
from .data.dataset import ReIDDataset, IMAGENET_MEAN, IMAGENET_STD
from .features import META_KEYS, load_cache, load_inference_model
from .official_eval import blended_embedding, normalize, write_candidates, write_embeddings, write_submission
from .pairs import build_cross_knn_context, load_reranker, pair_feature_np_cross, pair_features_np_cross_batch
from .refusal import refusal_probability
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


def _assert_pickle_free_npz(path: str | Path) -> None:
    with np.load(path, allow_pickle=False) as d:
        for k in d.files:
            _ = d[k]


def _collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = [x[k] for x in batch] if k in META_KEYS else torch.stack([x[k] for x in batch])
    return out


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype)[:, None, None]
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype)[:, None, None]
    y = (x * std + mean).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(y)


def _make_dataset_loader(manifest, image_size, prep, batch_size, workers):
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
    loader_kwargs = dict(
        batch_size=int(batch_size), shuffle=False, num_workers=int(workers),
        pin_memory=True, collate_fn=_collate,
    )
    if int(workers) > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(ds, **loader_kwargs)
    return df, loader


def _finalize_cache(df, acc: dict[str, list], out_npz: str | Path, *, write: bool = True):
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
    if write:
        out = Path(out_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **payload)
        _assert_pickle_free_npz(out)
    return payload


def _new_acc():
    return {k: [] for k in [
        "z_fused", "z_global", "z_local", "parts", "visibility", "visibility_score", "local", "color",
        "sample_id", "vehicle_key", "camera_id", "path",
    ]}


def _append_meta_and_color(acc, b, x_cpu: torch.Tensor) -> None:
    if "color" in b:
        acc["color"].extend(b["color"].float().cpu().numpy())
    else:
        for j in range(len(x_cpu)):
            acc["color"].append(robust_color_descriptor(_tensor_to_pil(x_cpu[j]), None))
    acc["sample_id"].extend(b["sample_id"])
    acc["vehicle_key"].extend(b["vehicle_key"])
    acc["camera_id"].extend(b["camera_id"])
    acc["path"].extend(b["path"])


def _stats_base(kind: str, n: int, batch_size: int, workers: int) -> dict:
    return {
        "runtime": kind,
        "samples": int(n),
        "batch_size": int(batch_size),
        "workers": int(workers),
        "runtime_init_s": 0.0,
        "dataset_loader_init_s": 0.0,
        "input_wait_preprocess_s": 0.0,
        "input_conversion_or_h2d_s": 0.0,
        "neural_forward_s": 0.0,
        "feature_postprocess_s": 0.0,
        "cache_write_s": 0.0,
        "total_s": 0.0,
    }


def extract_feature_cache_pt_timed(
    manifest, checkpoint, out_npz, *, device="auto", precision="bf16", batch_size=48,
    workers=8, image_size=None, write=True, warmup_batches=0,
):
    t_all = _now()
    t = _now()
    model, dev, mcfg = load_inference_model(checkpoint, device)
    _sync_cuda(dev)
    init_s = _now() - t
    prep = mcfg.get("preprocess", {}) or {}
    size = image_size if image_size is not None else prep.get("image_size", [256, 384])

    t = _now()
    df, loader = _make_dataset_loader(manifest, size, prep, batch_size, workers)
    loader_init_s = _now() - t
    stats = _stats_base("pytorch", len(df), batch_size, workers)
    stats.update({"runtime_init_s": init_s, "dataset_loader_init_s": loader_init_s, "device": str(dev), "precision": precision, "image_size": list(map(int, size))})
    acc = _new_acc()

    it = iter(loader)
    bi = 0
    while True:
        t = _now()
        try:
            b = next(it)
        except StopIteration:
            break
        stats["input_wait_preprocess_s"] += _now() - t
        x_cpu = b["image"]

        t = _now()
        x = x_cpu.to(dev, non_blocking=True)
        _sync_cuda(dev)
        stats["input_conversion_or_h2d_s"] += _now() - t

        if bi < int(warmup_batches):
            with torch.inference_mode(), autocast_context(dev, precision):
                _ = model(x)
            _sync_cuda(dev)

        t = _now()
        with torch.inference_mode(), autocast_context(dev, precision):
            o = model(x)
        _sync_cuda(dev)
        stats["neural_forward_s"] += _now() - t

        t = _now()
        acc["z_fused"].append(o["z_fused"].float().cpu().numpy())
        acc["z_global"].append(o["z_global"].float().cpu().numpy())
        acc["z_local"].append(o["z_local"].float().cpu().numpy())
        acc["parts"].append(o["parts"].float().cpu().numpy().astype(np.float16))
        acc["visibility"].append(o["visibility"].cpu().numpy().astype(np.uint8))
        acc["visibility_score"].append(o["visibility_score"].float().cpu().numpy().astype(np.float16))
        acc["local"].append(o["local"].float().cpu().numpy().astype(np.float16))
        _append_meta_and_color(acc, b, x_cpu)
        stats["feature_postprocess_s"] += _now() - t
        bi += 1

    t = _now()
    payload = _finalize_cache(df, acc, out_npz, write=write)
    stats["cache_write_s"] = _now() - t if write else 0.0
    stats["total_s"] = _now() - t_all
    stats["throughput_samples_s"] = float(len(df) / max(stats["total_s"], 1e-9))
    stats["neural_ms_per_sample"] = float(1000 * stats["neural_forward_s"] / max(len(df), 1))
    return payload, stats


def _import_ort():
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime is not installed. Install GPU runtime without sudo with: "
            "python -m pip install -U onnxruntime-gpu"
        ) from e
    return ort


def create_ort_session(onnx_path: str | Path, *, provider="auto", device="0", intra_threads=0, inter_threads=0):
    ort = _import_ort()
    available = ort.get_available_providers()
    p = str(provider).lower()
    if p == "auto":
        chosen = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else "CPUExecutionProvider"
    elif p in {"cuda", "gpu"}:
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable; available={available}")
        chosen = "CUDAExecutionProvider"
    elif p == "cpu":
        chosen = "CPUExecutionProvider"
    else:
        raise ValueError("provider must be auto/cuda/cpu")

    so = ort.SessionOptions()
    if int(intra_threads) > 0:
        so.intra_op_num_threads = int(intra_threads)
    if int(inter_threads) > 0:
        so.inter_op_num_threads = int(inter_threads)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers: list[Any]
    cuda_ep = "CUDAExecutionProvider" if int(device) == 0 else ("CUDAExecutionProvider", {"device_id": int(device)})
    if chosen == "CUDAExecutionProvider":
        # Device 0 is the CUDA EP default. Omitting an explicit device_id avoids the noisy
        # plugin-device warning emitted by recent onnxruntime-gpu builds.
        providers = [cuda_ep, "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
    active = sess.get_providers()
    if chosen == "CUDAExecutionProvider" and "CUDAExecutionProvider" not in active:
        raise RuntimeError(f"CUDA EP requested but session providers are {active}")
    return sess, chosen, available


def _ort_numpy_dtype(type_string: str):
    t = str(type_string).lower()
    if t == "tensor(float)":
        return np.float32
    if t == "tensor(float16)":
        return np.float16
    if t == "tensor(double)":
        return np.float64
    if "bfloat16" in t:
        raise RuntimeError(
            "This ONNX model uses BF16 input. NumPy/ORT Python feed support is not portable here; "
            "export FP16 or FP32 ONNX for folder inference."
        )
    raise RuntimeError(f"Unsupported ONNX input type: {type_string}")


def read_onnx_manifest(onnx_dir: str | Path) -> dict:
    p = Path(onnx_dir) / "onnx_manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def extract_feature_cache_onnx_timed(
    manifest, onnx_path, out_npz, *, onnx_manifest=None, provider="auto", device="0",
    batch_size=48, workers=8, image_size=None, write=True, warmup_batches=0,
):
    t_all = _now()
    t = _now()
    sess, chosen_provider, available = create_ort_session(onnx_path, provider=provider, device=device)
    init_s = _now() - t
    inp = sess.get_inputs()[0]
    input_name = inp.name
    input_dtype = _ort_numpy_dtype(inp.type)
    output_names = [x.name for x in sess.get_outputs()]
    miss = [x for x in FULL_OUTPUTS if x not in output_names]
    if miss:
        raise RuntimeError(f"ONNX model missing required outputs {miss}; has {output_names}")

    mf = onnx_manifest or {}
    size = image_size or mf.get("image_size") or [384, 576]
    prep = dict(mf.get("preprocess", {}) or {})
    prep.setdefault("augmentation_profile", "baseline_v4")
    prep.setdefault("use_source_bbox", True)
    prep.setdefault("bbox_pad", .03)
    t = _now()
    df, loader = _make_dataset_loader(manifest, size, prep, batch_size, workers)
    loader_init_s = _now() - t
    stats = _stats_base("onnxruntime", len(df), batch_size, workers)
    stats.update({
        "runtime_init_s": init_s, "dataset_loader_init_s": loader_init_s,
        "provider": chosen_provider, "available_providers": available,
        "input_type": inp.type, "image_size": list(map(int, size)),
    })
    acc = _new_acc()
    it = iter(loader)
    bi = 0
    while True:
        t = _now()
        try:
            b = next(it)
        except StopIteration:
            break
        stats["input_wait_preprocess_s"] += _now() - t
        x_cpu = b["image"]

        t = _now()
        x_np = np.ascontiguousarray(x_cpu.numpy().astype(input_dtype, copy=False))
        stats["input_conversion_or_h2d_s"] += _now() - t

        if bi < int(warmup_batches):
            _ = sess.run(list(FULL_OUTPUTS), {input_name: x_np})

        t = _now()
        outs = sess.run(list(FULL_OUTPUTS), {input_name: x_np})
        stats["neural_forward_s"] += _now() - t

        t = _now()
        o = {k: v for k, v in zip(FULL_OUTPUTS, outs)}
        acc["z_fused"].append(np.asarray(o["z_fused"], np.float32))
        acc["z_global"].append(np.asarray(o["z_global"], np.float32))
        acc["z_local"].append(np.asarray(o["z_local"], np.float32))
        acc["parts"].append(np.asarray(o["parts"], np.float16))
        acc["visibility"].append((np.asarray(o["visibility"]) > .5).astype(np.uint8))
        acc["visibility_score"].append(np.asarray(o["visibility_score"], np.float16))
        acc["local"].append(np.asarray(o["local"], np.float16))
        _append_meta_and_color(acc, b, x_cpu)
        stats["feature_postprocess_s"] += _now() - t
        bi += 1

    t = _now()
    payload = _finalize_cache(df, acc, out_npz, write=write)
    stats["cache_write_s"] = _now() - t if write else 0.0
    stats["total_s"] = _now() - t_all
    stats["throughput_samples_s"] = float(len(df) / max(stats["total_s"], 1e-9))
    stats["neural_ms_per_sample"] = float(1000 * stats["neural_forward_s"] / max(len(df), 1))
    return payload, stats


def extract_feature_caches_multi_onnx_timed(
    manifest,
    model_specs: list[dict],
    *,
    provider="cuda",
    device="0",
    batch_size=32,
    workers=8,
    write=False,
    out_paths: dict[str, str | Path] | None = None,
    warmup_batches=0,
):
    """Extract several ONNX feature models from one decoded/preprocessed image stream.

    Production uses this for ``base_full + external_convnext_full`` so JPEG decode,
    BBox crop, resize/normalize and the color descriptor are performed once per object,
    not once per ensemble member. The two ONNX sessions still execute independently.
    """
    if len(model_specs) < 2:
        raise ValueError("multi ONNX extraction needs at least two model specs")

    t_all = _now()
    sessions = []
    sizes = []
    preps = []
    init_total = 0.0
    input_types = []
    for spec in model_specs:
        name = str(spec["name"])
        onnx_path = Path(spec["onnx_path"])
        mf = dict(spec.get("onnx_manifest") or {})
        t = _now()
        sess, chosen_provider, available = create_ort_session(
            onnx_path, provider=provider, device=device
        )
        init_s = _now() - t
        init_total += init_s
        inp = sess.get_inputs()[0]
        output_names = [x.name for x in sess.get_outputs()]
        miss = [x for x in FULL_OUTPUTS if x not in output_names]
        if miss:
            raise RuntimeError(f"{name}: ONNX model missing outputs {miss}; has {output_names}")
        size = tuple(map(int, mf.get("image_size") or [384, 576]))
        prep = dict(mf.get("preprocess", {}) or {})
        prep.setdefault("augmentation_profile", "baseline_v4")
        prep.setdefault("use_source_bbox", True)
        prep.setdefault("bbox_pad", .03)
        sessions.append({
            "name": name,
            "session": sess,
            "input_name": inp.name,
            "input_dtype": _ort_numpy_dtype(inp.type),
            "input_type": inp.type,
            "provider": chosen_provider,
            "available_providers": available,
            "init_s": init_s,
            "onnx_path": str(onnx_path),
        })
        sizes.append(size)
        preps.append(prep)
        input_types.append(inp.type)

    if len(set(sizes)) != 1:
        raise RuntimeError(f"Ensemble ONNX image sizes differ: {sizes}; shared preprocessing is unsafe")
    prep_keys = ("augmentation_profile", "use_source_bbox", "bbox_pad")
    prep_sig = [tuple(p.get(k) for k in prep_keys) for p in preps]
    if len(set(prep_sig)) != 1:
        raise RuntimeError(f"Ensemble preprocessing differs: {prep_sig}; shared preprocessing is unsafe")
    if len(set(input_types)) != 1:
        raise RuntimeError(f"Ensemble ONNX input dtypes differ: {input_types}")

    t = _now()
    df, loader = _make_dataset_loader(manifest, sizes[0], preps[0], batch_size, workers)
    loader_init_s = _now() - t
    shared = {
        "runtime": "onnxruntime_multi",
        "samples": int(len(df)),
        "members": [x["name"] for x in sessions],
        "batch_size": int(batch_size),
        "workers": int(workers),
        "runtime_init_s": float(init_total),
        "dataset_loader_init_s": float(loader_init_s),
        "input_wait_preprocess_s": 0.0,
        "input_conversion_or_h2d_s": 0.0,
        "feature_postprocess_s": 0.0,
        "cache_write_s": 0.0,
        "total_s": 0.0,
        "image_size": list(sizes[0]),
    }
    member_stats = {
        x["name"]: {
            "name": x["name"],
            "provider": x["provider"],
            "available_providers": x["available_providers"],
            "input_type": x["input_type"],
            "runtime_init_s": float(x["init_s"]),
            "neural_forward_s": 0.0,
            "neural_ms_per_sample": 0.0,
            "onnx_path": x["onnx_path"],
        }
        for x in sessions
    }
    accs = {x["name"]: _new_acc() for x in sessions}
    it = iter(loader)
    bi = 0
    while True:
        t = _now()
        try:
            b = next(it)
        except StopIteration:
            break
        shared["input_wait_preprocess_s"] += _now() - t
        x_cpu = b["image"]
        t = _now()
        input_dtype = sessions[0]["input_dtype"]
        x_np = np.ascontiguousarray(x_cpu.numpy().astype(input_dtype, copy=False))
        shared["input_conversion_or_h2d_s"] += _now() - t

        if bi < int(warmup_batches):
            for info in sessions:
                _ = info["session"].run(list(FULL_OUTPUTS), {info["input_name"]: x_np})

        outputs = {}
        for info in sessions:
            t = _now()
            outs = info["session"].run(list(FULL_OUTPUTS), {info["input_name"]: x_np})
            member_stats[info["name"]]["neural_forward_s"] += _now() - t
            outputs[info["name"]] = outs

        t = _now()
        for info in sessions:
            acc = accs[info["name"]]
            o = {k: v for k, v in zip(FULL_OUTPUTS, outputs[info["name"]])}
            acc["z_fused"].append(np.asarray(o["z_fused"], np.float32))
            acc["z_global"].append(np.asarray(o["z_global"], np.float32))
            acc["z_local"].append(np.asarray(o["z_local"], np.float32))
            acc["parts"].append(np.asarray(o["parts"], np.float16))
            acc["visibility"].append((np.asarray(o["visibility"]) > .5).astype(np.uint8))
            acc["visibility_score"].append(np.asarray(o["visibility_score"], np.float16))
            acc["local"].append(np.asarray(o["local"], np.float16))
            _append_meta_and_color(acc, b, x_cpu)
        shared["feature_postprocess_s"] += _now() - t
        bi += 1

    payloads = {}
    t = _now()
    for name, acc in accs.items():
        out_path = (out_paths or {}).get(name, Path("unused.npz"))
        payloads[name] = _finalize_cache(df, acc, out_path, write=bool(write))
    shared["cache_write_s"] = _now() - t if write else 0.0
    shared["total_s"] = _now() - t_all
    shared["throughput_samples_s"] = float(len(df) / max(shared["total_s"], 1e-9))
    for info in sessions:
        ms = member_stats[info["name"]]
        ms["neural_ms_per_sample"] = float(1000 * ms["neural_forward_s"] / max(len(df), 1))
    shared["members_timing"] = list(member_stats.values())
    return payloads, shared


class PTRerankerBackend:
    def __init__(self, checkpoint: str | Path, *, device="cpu"):
        self.device = torch.device(device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu")
        t = _now()
        self.model = load_reranker(checkpoint, str(self.device))
        _sync_cuda(self.device)
        self.init_s = _now() - t
        self.name = "pytorch"

    def predict(self, feats: np.ndarray) -> np.ndarray:
        if len(feats) == 0:
            return np.zeros(0, np.float32)
        xb = torch.from_numpy(np.asarray(feats, np.float32)).to(self.device)
        _sync_cuda(self.device)
        with torch.inference_mode():
            y = torch.sigmoid(self.model(xb)).float().cpu().numpy()
        _sync_cuda(self.device)
        return np.asarray(y, np.float32)


class ONNXRerankerBackend:
    def __init__(self, onnx_path: str | Path, *, provider="auto", device="0"):
        t = _now()
        self.session, self.provider, self.available_providers = create_ort_session(onnx_path, provider=provider, device=device)
        self.init_s = _now() - t
        self.input = self.session.get_inputs()[0]
        self.input_name = self.input.name
        self.input_dtype = _ort_numpy_dtype(self.input.type)
        self.output_name = self.session.get_outputs()[0].name
        self.name = "onnxruntime"

    def predict(self, feats: np.ndarray) -> np.ndarray:
        if len(feats) == 0:
            return np.zeros(0, np.float32)
        x = np.ascontiguousarray(np.asarray(feats).astype(self.input_dtype, copy=False))
        y = self.session.run([self.output_name], {self.input_name: x})[0]
        return np.asarray(y, np.float32).reshape(-1)


def _cache_ids(cache: dict, requested: str | None = None):
    col = requested or ("image_id" if "meta_image_id" in cache else "sample_id")
    if col == "sample_id":
        return cache["sample_id"].astype(str)
    key = f"meta_{col}"
    if key not in cache:
        raise KeyError(f"{col} was not stored in cache")
    return cache[key].astype(str)


def _rank_caches_backend(q: dict, g: dict, recipe: dict, reranker_backend=None):
    timings = {
        "embedding_blend_similarity_s": 0.0,
        "neighbor_context_s": 0.0,
        "base_sort_filter_s": 0.0,
        "kreciprocal_jaccard_s": 0.0,
        "pair_feature_build_s": 0.0,
        "reranker_neural_s": 0.0,
        "reranker_merge_sort_s": 0.0,
    }
    alpha = float(recipe.get("base_alpha", 0.0))
    beta = float(np.clip(recipe.get("reranker_beta", 0.0), 0, 1))
    kl = float(np.clip(recipe.get("kreciprocal_lambda", 0.0), 0, 1))
    rk = int(recipe.get("rerank_topk", 100))
    kk = int(recipe.get("kreciprocal_k", 20))
    same_camera_filter = bool(recipe.get("same_camera_filter", False))

    t = _now()
    qe = blended_embedding(q["z_global"], q["z_fused"], alpha)
    ge = blended_embedding(g["z_global"], g["z_fused"], alpha)
    sim = normalize(qe) @ normalize(ge).T
    qids = _cache_ids(q); gids = _cache_ids(g)
    timings["embedding_blend_similarity_s"] += _now() - t

    need_knn = kl > 0 or (reranker_backend is not None and beta > 0)
    t = _now()
    qknn, gknn = build_cross_knn_context(q, g, kk) if need_knn else (None, None)
    timings["neighbor_context_s"] += _now() - t

    ranked, ranked_scores = {}, {}
    qcams = q.get("camera_id"); gcams = g.get("camera_id")
    camera_filter_available = False
    if same_camera_filter and qcams is not None and gcams is not None:
        qall = {str(x) for x in np.asarray(qcams).tolist()}; gall = {str(x) for x in np.asarray(gcams).tolist()}
        missing_tokens = {"-1", "", "none", "nan", "unknown"}
        camera_filter_available = bool((qall - missing_tokens) and (gall - missing_tokens))

    for qi, qid in enumerate(qids.tolist()):
        t = _now()
        base_order = np.argsort(-sim[qi], kind="stable")
        if camera_filter_available:
            qc = str(qcams[qi])
            base_order = np.asarray([gi for gi in base_order if str(gcams[int(gi)]) != qc], dtype=np.int64)
        order = base_order
        base01 = np.clip((sim[qi, base_order] + 1.0) * .5, 0, 1).astype(np.float32)
        pre = base01.copy()
        timings["base_sort_filter_s"] += _now() - t

        if kl > 0 and len(base_order):
            t = _now()
            jac = np.asarray([
                len(qknn[qi] & gknn[int(gi)]) / max(1, len(qknn[qi] | gknn[int(gi)]))
                for gi in base_order
            ], np.float32)
            pre = (1.0 - kl) * pre + kl * jac
            ro = np.argsort(-pre, kind="stable"); order = base_order[ro]; pre = pre[ro]
            timings["kreciprocal_jaccard_s"] += _now() - t
        scores = pre

        if reranker_backend is not None and beta > 0 and len(order):
            k = min(rk, len(order)); head = order[:k]
            t = _now()
            if qknn is not None and gknn is not None:
                aa = qknn[qi]; jac = []
                for gi in head:
                    bb = gknn[int(gi)]; u = len(aa | bb)
                    jac.append(float(len(aa & bb) / u) if u else 0.0)
            else:
                jac = None
            feats = pair_features_np_cross_batch(q, qi, g, head, neighbor_jaccard=jac)
            timings["pair_feature_build_s"] += _now() - t

            t = _now()
            rrs = reranker_backend.predict(feats)
            timings["reranker_neural_s"] += _now() - t

            t = _now()
            head_base = np.asarray([scores[np.flatnonzero(order == gi)[0]] for gi in head], np.float32)
            comb = (1.0 - beta) * head_base + beta * rrs
            ro = np.argsort(-comb, kind="stable"); head2 = head[ro]; score2 = comb[ro]
            tail = order[k:]; tail_score = scores[k:]
            order = np.concatenate([head2, tail]); scores = np.concatenate([score2, tail_score])
            timings["reranker_merge_sort_s"] += _now() - t

        ranked[str(qid)] = [str(gids[i]) for i in order]
        ranked_scores[str(qid)] = scores
    return ranked, ranked_scores, qe, ge, timings


def run_retrieval_backend_timed(
    query_cache, gallery_cache, out_dir, *, recipe: dict, refusal: dict | None,
    reranker_backend=None, query_id_column=None, gallery_id_column=None,
    output_topk=10, write=True,
):
    t_all = _now()
    t = _now(); q = load_cache(query_cache) if not isinstance(query_cache, dict) else query_cache; g = load_cache(gallery_cache) if not isinstance(gallery_cache, dict) else gallery_cache
    cache_load_s = _now() - t
    ranked, ranked_scores, qe, ge, timings = _rank_caches_backend(q, g, recipe, reranker_backend)
    timings["cache_load_s"] = cache_load_s

    t = _now()
    qids = _cache_ids(q, query_id_column); gids = _cache_ids(g, gallery_id_column)
    cache_qids = q["meta_image_id"].astype(str) if "meta_image_id" in q else q["sample_id"].astype(str)
    cache_gids = g["meta_image_id"].astype(str) if "meta_image_id" in g else g["sample_id"].astype(str)
    qmap = {str(a): str(b) for a, b in zip(cache_qids, qids)}
    gmap = {str(a): str(b) for a, b in zip(cache_gids, gids)}
    ranked_out = {qmap.get(k, k): [gmap.get(x, x) for x in v] for k, v in ranked.items()}
    map_s = _now() - t

    t = _now()
    candidate_rows = []
    qglob = normalize(q["z_global"]); gglob = normalize(g["z_global"])
    cache_gid_index = {str(x): i for i, x in enumerate(cache_gids)}
    for qi, cqid in enumerate(cache_qids):
        order = ranked.get(str(cqid), [])
        scores = ranked_scores.get(str(cqid), np.zeros(0, np.float32))
        accepted = bool(len(order)); match_prob = float(scores[0]) if len(scores) else 0.0
        if refusal and len(order):
            top_gid = str(order[0]); gi = cache_gid_index[top_gid]
            s1 = float(scores[0]); s2 = float(scores[1]) if len(scores) > 1 else 0.0
            top5 = float(np.mean(scores[:min(5, len(scores))])); glob = float(qglob[qi] @ gglob[gi])
            X = np.asarray([[s1, s1-s2, glob, s1-top5]], np.float32)
            match_prob = float(refusal_probability(refusal, X)[0]); accepted = match_prob >= float(refusal["threshold"])
        elif refusal:
            accepted = False
        if accepted and order:
            candidate_rows.append((qmap.get(str(cqid), str(cqid)), gmap.get(str(order[0]), str(order[0])), float(match_prob)))
    refusal_s = _now() - t

    write_s = 0.0
    if write:
        t = _now()
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        write_submission(out / "submission.csv", ranked_out, [str(x) for x in qids], top_k=output_topk)
        write_candidates(out / "candidates.csv", candidate_rows)
        write_embeddings(out / "embeddings.npy", qe, ge)
        (out / "retrieval_recipe_used.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
        write_s = _now() - t

    timings.update({
        "id_mapping_s": map_s,
        "refusal_s": refusal_s,
        "output_write_s": write_s,
        "total_retrieval_s": _now() - t_all,
        "queries": int(len(qids)),
        "gallery": int(len(gids)),
    })
    return {
        "ranked": ranked_out,
        "ranked_internal": ranked,
        "ranked_scores": ranked_scores,
        "qe": qe,
        "ge": ge,
        "candidate_rows": candidate_rows,
        "timings": timings,
    }


def cache_equivalence(a: dict | str | Path, b: dict | str | Path) -> dict:
    a = load_cache(a) if not isinstance(a, dict) else a
    b = load_cache(b) if not isinstance(b, dict) else b
    out = {}
    for k in ("z_global", "z_fused", "z_local"):
        x = np.asarray(a[k], np.float32); y = np.asarray(b[k], np.float32)
        if x.shape != y.shape:
            out[k] = {"shape_a": list(x.shape), "shape_b": list(y.shape), "compatible": False}
            continue
        cos = np.sum(normalize(x) * normalize(y), axis=1)
        out[k] = {
            "compatible": True,
            "max_abs": float(np.max(np.abs(x-y))),
            "mean_abs": float(np.mean(np.abs(x-y))),
            "mean_cosine": float(np.mean(cos)),
            "min_cosine": float(np.min(cos)),
        }
    return out


def ranking_agreement(a: dict, b: dict, topk: int = 10) -> dict:
    common = sorted(set(a) & set(b))
    if not common:
        return {"queries": 0}
    exact = 0; top1 = 0; jacc = []
    for q in common:
        aa = list(a[q])[:topk]; bb = list(b[q])[:topk]
        exact += int(aa == bb)
        top1 += int(bool(aa) and bool(bb) and aa[0] == bb[0])
        sa, sb = set(aa), set(bb); jacc.append(len(sa & sb) / max(1, len(sa | sb)))
    n = len(common)
    return {
        "queries": n,
        "top1_agreement": float(top1/n),
        "top10_exact_agreement": float(exact/n),
        "top10_jaccard_mean": float(np.mean(jacc)),
    }
