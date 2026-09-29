#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.data.color import robust_color_descriptor
from vehicle_fingerprint.data.crop import crop_bbox
from vehicle_fingerprint.data.dataset import _baseline_v4_transform, _crop_source_bbox
from vehicle_fingerprint.pairs import build_cross_knn_context, pair_feature_np_cross
from vehicle_fingerprint.refusal import refusal_probability
from vehicle_fingerprint.tensorrt_deploy import TensorRTEngine, summarize_times


def _rooted(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _portable_path(export_dir: Path, manifest_value, fallback: Path) -> Path:
    if manifest_value:
        p = Path(str(manifest_value))
        if p.is_file():
            return p
        if not p.is_absolute() and (ROOT / p).is_file():
            return ROOT / p
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Required deployment file not found: {manifest_value!r}; fallback={fallback}")


def _open_row(row) -> Image.Image:
    idx = row.index
    source = _rooted(row.source_path) if "source_path" in idx and str(row.source_path) not in ("", "nan", "None") else None
    can_bbox = source is not None and source.is_file() and all(k in idx for k in ("x", "y", "w", "h"))
    path = source if can_bbox else _rooted(row.path if "path" in idx else row.image_path)
    with Image.open(path) as im:
        im = im.convert("RGB")
        if can_bbox:
            im = _crop_source_bbox(im, row, 0.03)
        return im.copy()


def _recipe_embedding_torch(zg: torch.Tensor, zf: torch.Tensor, alpha: float) -> torch.Tensor:
    a = float(np.clip(alpha, 0.0, 1.0))
    zg = torch.nn.functional.normalize(zg.float(), dim=-1)
    zf = torch.nn.functional.normalize(zf.float(), dim=-1)
    if a <= 0:
        return zg
    if a >= 1:
        return zf
    return torch.cat([math.sqrt(1.0-a)*zg, math.sqrt(a)*zf], dim=-1)


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cache)


def _load_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _build_gallery_cache(full: TensorRTEngine, manifest: Path, cache_path: Path, *, h: int, w: int, batch: int) -> dict:
    df = pd.read_csv(manifest)
    transform = _baseline_v4_transform((h, w), False)
    tensors, colors, ids, cameras = [], [], [], []
    t0 = time.perf_counter()
    for i, row in df.iterrows():
        im = _open_row(row)
        tensors.append(transform(im))
        colors.append(robust_color_descriptor(im, None))
        ids.append(str(row.image_id) if "image_id" in row.index else str(row.sample_id if "sample_id" in row.index else i))
        cameras.append(str(row.camera_id) if "camera_id" in row.index else "-1")
    prep_s = time.perf_counter() - t0
    outs = {k: [] for k in ("z_fused", "z_global", "z_local", "parts", "visibility", "visibility_score", "local")}
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for st in range(0, len(tensors), int(batch)):
        x = torch.stack(tensors[st:st+int(batch)]).to(full.device)
        o = full.infer(x); torch.cuda.synchronize()
        for k in outs:
            outs[k].append(o[k].detach().float().cpu().numpy())
    infer_s = time.perf_counter() - t0
    cache = {k: np.concatenate(v, axis=0) for k, v in outs.items()}
    cache["visibility"] = (cache["visibility"] >= 0.5).astype(np.uint8)
    cache["color"] = np.stack(colors).astype(np.float32)
    cache["camera_id"] = np.asarray(cameras, dtype=str)
    cache["sample_id"] = np.asarray(ids, dtype=str)
    _save_cache(cache_path, cache)
    print(json.dumps({"gallery_cache": str(cache_path), "images": len(ids), "preprocess_s": prep_s, "tensorrt_s": infer_s, "total_s": prep_s+infer_s}, indent=2))
    return cache


def _cuda_ms(fn):
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record(); out = fn(); en.record(); en.synchronize()
    return out, float(st.elapsed_time(en))


def _infer_once(*, query_path: Path, bbox, full: TensorRTEngine, rer: TensorRTEngine | None, gallery: dict, recipe: dict, refusal: dict | None, h: int, w: int, topk: int):
    total0 = time.perf_counter()
    t = time.perf_counter()
    with Image.open(query_path) as src:
        im = src.convert("RGB")
        if bbox is not None:
            im = crop_bbox(im, *bbox, pad=0.03)
        im = im.copy()
    transform = _baseline_v4_transform((h, w), False)
    color = robust_color_descriptor(im, None)[None].astype(np.float32)
    x = transform(im)[None]
    preprocess_ms = (time.perf_counter() - t) * 1000.0

    torch.cuda.synchronize(); t = time.perf_counter(); x = x.to(full.device, non_blocking=False); torch.cuda.synchronize()
    h2d_ms = (time.perf_counter() - t) * 1000.0
    o, feature_ms = _cuda_ms(lambda: full.infer(x))

    q = {k: o[k].detach().float().cpu().numpy() for k in ("z_fused", "z_global", "z_local", "parts", "visibility", "visibility_score", "local")}
    q["visibility"] = (q["visibility"] >= 0.5).astype(np.uint8)
    q["color"] = color; q["camera_id"] = np.asarray(["-1"]); q["sample_id"] = np.asarray([query_path.name])

    alpha = float(recipe.get("base_alpha", 0.0)); beta = float(recipe.get("reranker_beta", 0.0))
    kl = float(recipe.get("kreciprocal_lambda", 0.0)); kk = int(recipe.get("kreciprocal_k", 20)); rk = int(recipe.get("rerank_topk", 100))
    gg = torch.from_numpy(gallery["z_global"]).to(full.device); gf = torch.from_numpy(gallery["z_fused"]).to(full.device)
    gallery_z = _recipe_embedding_torch(gg, gf, alpha); qz = _recipe_embedding_torch(o["z_global"], o["z_fused"], alpha)
    (base_sim,), ann_ms = _cuda_ms(lambda: (qz @ gallery_z.T,))
    sim = base_sim[0].detach().float().cpu().numpy()
    base_order = np.argsort(-sim, kind="stable")
    base01 = np.clip((sim[base_order] + 1.0) * 0.5, 0, 1).astype(np.float32)

    t = time.perf_counter(); qknn = gknn = None
    pre = base01.copy(); order = base_order
    if kl > 0:
        qknn, gknn = build_cross_knn_context(q, gallery, kk)
        jac = np.asarray([len(qknn[0] & gknn[int(gi)]) / max(1, len(qknn[0] | gknn[int(gi)])) for gi in base_order], np.float32)
        pre = (1.0-kl) * pre + kl * jac
        ro = np.argsort(-pre, kind="stable"); order = base_order[ro]; pre = pre[ro]
    kreciprocal_ms = (time.perf_counter() - t) * 1000.0

    scores = pre
    pair_ms = rer_ms = 0.0
    if rer is not None and beta > 0 and len(order):
        k = min(rk, len(order)); head = order[:k]
        t = time.perf_counter()
        feats = []
        for gi in head:
            nj = 0.0
            if qknn is not None:
                a_set, b_set = qknn[0], gknn[int(gi)]
                nj = len(a_set & b_set) / max(1, len(a_set | b_set))
            feats.append(pair_feature_np_cross(q, 0, gallery, int(gi), neighbor_jaccard=float(nj)))
        X = np.stack(feats).astype(np.float32)
        pair_ms = (time.perf_counter() - t) * 1000.0
        xt = torch.from_numpy(X).to(full.device)
        ro, rer_ms = _cuda_ms(lambda: rer.infer(xt))
        prob = ro.get("match_probability", next(iter(ro.values()))).detach().float().cpu().numpy().reshape(-1)
        comb = (1.0-beta) * scores[:k] + beta * prob
        rr_order = np.argsort(-comb, kind="stable")
        order = np.concatenate([head[rr_order], order[k:]])
        scores = np.concatenate([comb[rr_order], scores[k:]])

    t = time.perf_counter(); accepted = bool(len(order)); match_prob = float(scores[0]) if len(scores) else 0.0
    if refusal and len(order):
        s1 = float(scores[0]); s2 = float(scores[1]) if len(scores) > 1 else 0.0; top5 = float(np.mean(scores[:min(5, len(scores))]))
        gi = int(order[0]); qg = q["z_global"][0].astype(np.float32); gg0 = gallery["z_global"][gi].astype(np.float32)
        qg /= max(float(np.linalg.norm(qg)), 1e-12); gg0 /= max(float(np.linalg.norm(gg0)), 1e-12)
        glob = float(qg @ gg0)
        Xr = np.asarray([[s1, s1-s2, glob, s1-top5]], np.float32)
        match_prob = float(refusal_probability(refusal, Xr)[0]); accepted = match_prob >= float(refusal["threshold"])
    refusal_ms = (time.perf_counter() - t) * 1000.0
    total_ms = (time.perf_counter() - total0) * 1000.0
    ids = gallery["sample_id"].astype(str)
    candidates = [{"id": str(ids[int(gi)]), "score": float(scores[i])} for i, gi in enumerate(order[:int(topk)])]
    return {
        "accepted": accepted, "match_probability": match_prob, "candidates": candidates,
        "timing_ms": {"preprocess_decode_crop_resize": preprocess_ms, "h2d": h2d_ms, "feature_trt": feature_ms, "ann_similarity": ann_ms, "kreciprocal_cpu": kreciprocal_ms, "pair_features_cpu": pair_ms, "reranker_trt": rer_ms, "refusal_cpu": refusal_ms, "end_to_end": total_ms},
    }


def main():
    ap = argparse.ArgumentParser(description="Exact v0.9.4 TensorRT single-image inference with stage latency breakdown")
    ap.add_argument("--query-image", required=True)
    ap.add_argument("--gallery-manifest", required=True)
    ap.add_argument("--gallery-cache", default="artifacts/tensorrt_v094/gallery_full_cache.npz")
    ap.add_argument("--rebuild-gallery", action="store_true")
    ap.add_argument("--export-dir", default="deploy/tensorrt_v094")
    ap.add_argument("--deployment-dir", default="deploy/models_v094_v5exact_base_veri")
    ap.add_argument("--device", default="0"); ap.add_argument("--gallery-batch", type=int, default=8)
    ap.add_argument("--topk", type=int, default=10); ap.add_argument("--warmup", type=int, default=10); ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("X", "Y", "W", "H"), default=None)
    ap.add_argument("--out", default="artifacts/benchmarks/tensorrt_v094/single_inference.json")
    a = ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA is required")
    device = torch.device(f"cuda:{a.device}"); torch.cuda.set_device(device)
    export_dir = _rooted(a.export_dir); dep_dir = _rooted(a.deployment_dir)
    manifest = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8")); models = manifest["models"]
    h = int(manifest["shape_profile"]["height"]); w = int(manifest["shape_profile"]["width"])
    fc = models.get("full_feature_extractor", {})
    full_path = _portable_path(export_dir, fc.get("engine"), export_dir / "engines" / "full_feature_extractor.plan")
    full = TensorRTEngine(full_path, device)
    rer = None
    rc = models.get("pair_reranker", {})
    rer_fallback = export_dir / "engines" / "pair_reranker.plan"
    if rc.get("engine") or rer_fallback.is_file(): rer = TensorRTEngine(_portable_path(export_dir, rc.get("engine"), rer_fallback), device)
    recipe_path = _portable_path(export_dir, fc.get("retrieval_recipe"), dep_dir / "retrieval_recipe.json")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    refusal = None
    refusal_fallback = dep_dir / "refusal.json"
    try: refusal_path = _portable_path(export_dir, fc.get("refusal"), refusal_fallback); refusal = json.loads(refusal_path.read_text(encoding="utf-8"))
    except FileNotFoundError: pass

    cache_path = _rooted(a.gallery_cache); gallery_manifest = _rooted(a.gallery_manifest)
    gallery = _build_gallery_cache(full, gallery_manifest, cache_path, h=h, w=w, batch=a.gallery_batch) if a.rebuild_gallery or not cache_path.is_file() else _load_cache(cache_path)
    query = _rooted(a.query_image)
    # Warm up neural engines without reporting their first-run setup costs.
    transform = _baseline_v4_transform((h, w), False)
    with Image.open(query) as im: wx = transform(im.convert("RGB"))[None].to(device)
    for _ in range(max(0, int(a.warmup))):
        oo = full.infer(wx)
        if rer is not None:
            dummy = torch.zeros((min(100, len(gallery["sample_id"])), 33), device=device, dtype=torch.float16); rer.infer(dummy)
    torch.cuda.synchronize()

    runs = [_infer_once(query_path=query, bbox=a.bbox, full=full, rer=rer, gallery=gallery, recipe=recipe, refusal=refusal, h=h, w=w, topk=a.topk) for _ in range(max(1, int(a.repeat)))]
    stages = runs[0]["timing_ms"].keys(); timing = {k: summarize_times([r["timing_ms"][k] for r in runs]) for k in stages}
    result = {"query_image": str(query), "gallery_images": int(len(gallery["sample_id"])), "recipe": recipe, "result": {k:v for k,v in runs[-1].items() if k != "timing_ms"}, "timing": timing, "repeat": len(runs), "gpu": torch.cuda.get_device_name(device)}
    out = _rooted(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2)); print(f"\n[DONE] {out}")


if __name__ == "__main__":
    main()
