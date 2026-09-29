from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from .features import load_cache
from .pairs import pair_feature_np, load_reranker
from .metrics import refusal_metrics


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    return x


def _select_query_indices(indices: np.ndarray, cams: np.ndarray, rng: np.random.Generator, max_queries_per_id: int) -> list[int]:
    """Select camera-diverse queries so identities with many frames do not dominate metrics."""
    by_cam: dict[str, list[int]] = {}
    for i in indices.tolist():
        by_cam.setdefault(str(cams[i]), []).append(int(i))
    cam_names = list(by_cam)
    rng.shuffle(cam_names)
    chosen = []
    for cam in cam_names[:max(1, int(max_queries_per_id))]:
        chosen.append(int(rng.choice(by_cam[cam])))
    return chosen


def build_open_set_protocol(
    c: dict[str, np.ndarray],
    *,
    seed: int = 42,
    known_fraction: float = 0.60,
    max_queries_per_id: int = 2,
) -> dict:
    """Build a deterministic open-set gallery/query protocol from a held-out identity split.

    Known identities are placed in the gallery and queried from camera-diverse images. Unknown
    identities contribute queries only, so a correct system must refuse them. Known identities
    must have >=2 cameras so a genuine cross-camera match can exist.
    """
    keys = c["vehicle_key"].astype(str)
    cams = c["camera_id"].astype(str)
    uniq = np.array(sorted(set(keys)))
    rng = np.random.default_rng(seed)

    eligible_known = []
    unknown_candidates = []
    for k in uniq:
        inds = np.flatnonzero(keys == k)
        ncam = len(np.unique(cams[inds]))
        if ncam >= 2:
            eligible_known.append(k)
        unknown_candidates.append(k)

    eligible_known = np.asarray(eligible_known, dtype=str)
    if len(eligible_known) < 2:
        raise RuntimeError("Open-set protocol needs at least two identities observed by >=2 cameras")
    rng.shuffle(eligible_known)
    n_known = int(round(len(eligible_known) * float(known_fraction)))
    n_known = min(max(1, n_known), len(eligible_known) - 1)
    known_ids = set(eligible_known[:n_known].tolist())

    # Unknown IDs are sampled from identities not in the gallery. Keep roughly comparable counts
    # to stabilize F1/TNR while preserving true identity disjointness between classes.
    unknown_pool = [k for k in unknown_candidates if k not in known_ids]
    rng.shuffle(unknown_pool)
    n_unknown = min(len(unknown_pool), max(1, len(known_ids)))
    unknown_ids = set(unknown_pool[:n_unknown])
    if not unknown_ids:
        raise RuntimeError("Open-set protocol could not create unknown identities")

    gallery = np.flatnonzero(np.isin(keys, list(known_ids))).astype(np.int64)
    queries: list[int] = []
    labels: list[int] = []

    for k in sorted(known_ids):
        inds = np.flatnonzero(keys == k)
        for qi in _select_query_indices(inds, cams, rng, max_queries_per_id):
            # Query remains valid only if gallery has another-camera positive.
            has_pos = np.any((keys[gallery] == k) & (cams[gallery] != cams[qi]) & (gallery != qi))
            if has_pos:
                queries.append(qi); labels.append(1)

    for k in sorted(unknown_ids):
        inds = np.flatnonzero(keys == k)
        for qi in _select_query_indices(inds, cams, rng, max_queries_per_id):
            queries.append(qi); labels.append(0)

    if len(set(labels)) < 2:
        raise RuntimeError("Open-set protocol must contain both known and unknown queries")
    return {
        "gallery": np.asarray(gallery, dtype=np.int64),
        "queries": np.asarray(queries, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.int64),
        "known_ids": sorted(known_ids),
        "unknown_ids": sorted(unknown_ids),
        "seed": int(seed),
        "known_fraction": float(known_fraction),
        "max_queries_per_id": int(max_queries_per_id),
    }


def _rerank_scores(c, qi: int, cand: np.ndarray, reranker, device: str, batch_size: int = 2048) -> np.ndarray:
    feats = np.stack([pair_feature_np(c, qi, int(gi)) for gi in cand]).astype(np.float32)
    dev = torch.device(device)
    out = []
    with torch.inference_mode():
        for s in range(0, len(feats), batch_size):
            xb = torch.from_numpy(feats[s:s + batch_size]).to(dev)
            out.append(torch.sigmoid(reranker(xb)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def build_refusal_features(
    cache_path: str | Path,
    *,
    reranker_path: str | Path | None = None,
    retrieval_recipe: dict | None = None,
    seed: int = 42,
    known_fraction: float = 0.60,
    max_queries_per_id: int = 2,
    ann_topk: int = 100,
    reranker_device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    c = load_cache(cache_path)
    z_fused = _normalize(c["z_fused"])
    z_global = _normalize(c["z_global"])
    recipe = retrieval_recipe or {}
    alpha = float(np.clip(recipe.get("base_alpha", 1.0), 0.0, 1.0))
    beta = float(np.clip(recipe.get("reranker_beta", 1.0 if reranker_path else 0.0), 0.0, 1.0))
    same_camera_policy = str(recipe.get("same_camera_policy", "same_identity"))
    keys = c["vehicle_key"].astype(str)
    cams = c["camera_id"].astype(str)
    sample_ids = c.get("sample_id", np.arange(len(keys)).astype(str)).astype(str)
    protocol = build_open_set_protocol(
        c, seed=seed, known_fraction=known_fraction, max_queries_per_id=max_queries_per_id
    )
    gallery = protocol["gallery"]
    queries = protocol["queries"]
    y = protocol["labels"]

    reranker = load_reranker(reranker_path, reranker_device) if reranker_path and beta > 0 else None
    X = []
    rows = []
    for row_i, (qi, target) in enumerate(zip(queries.tolist(), y.tolist())):
        # Same-ID/same-camera instances are junk for a known query. For unknown identities this
        # rule changes nothing because their identity is absent from the gallery by construction.
        valid = gallery != qi
        if same_camera_policy == "all":
            valid &= cams[gallery] != cams[qi]
        elif target == 1:
            valid &= ~((keys[gallery] == keys[qi]) & (cams[gallery] == cams[qi]))
        cand = gallery[valid]
        if len(cand) == 0:
            continue

        base_global = z_global[cand] @ z_global[qi]
        base_fused = z_fused[cand] @ z_fused[qi]
        base = (1.0 - alpha) * base_global + alpha * base_fused
        order = np.argsort(-base, kind="stable")[:min(int(ann_topk), len(cand))]
        top = cand[order]
        base_top = base[order].astype(np.float32)
        if reranker is not None:
            rr = _rerank_scores(c, qi, top, reranker, reranker_device)
            base01 = np.clip((base_top + 1.0) * 0.5, 0.0, 1.0)
            scores = (1.0 - beta) * base01 + beta * rr
            rr_order = np.argsort(-scores, kind="stable")
            top = top[rr_order]
            scores = scores[rr_order]
            base_top = base_top[rr_order]
        else:
            scores = np.clip((base_top + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

        if len(scores) == 0:
            continue
        s1 = float(scores[0])
        s2 = float(scores[1] if len(scores) > 1 else 0.0)
        global1 = float(z_global[qi] @ z_global[top[0]])
        top5mean = float(np.mean(scores[:min(5, len(scores))]))
        f = [s1, s1 - s2, global1, s1 - top5mean]
        X.append(f)
        rows.append({
            "query_sample_id": sample_ids[qi],
            "query_vehicle_id": keys[qi],
            "query_camera_id": cams[qi],
            "is_known": int(target),
            "top1_sample_id": sample_ids[top[0]],
            "top1_vehicle_id": keys[top[0]],
            "top1_camera_id": cams[top[0]],
            "top1_score": s1,
            "margin12": s1 - s2,
            "global_top1": global1,
            "top1_minus_top5mean": s1 - top5mean,
        })
    return np.asarray(X, np.float32), np.asarray([r["is_known"] for r in rows], np.int64), pd.DataFrame(rows), protocol


def fit_refusal_calibrator(
    cache_path: str | Path,
    out_json: str | Path,
    reranker_path: str | Path | None = None,
    retrieval_recipe: dict | None = None,
    seed: int = 42,
    known_fraction: float = 0.60,
    max_queries_per_id: int = 2,
    ann_topk: int = 100,
    reranker_device: str = "cpu",
):
    X, y, rows, protocol = build_refusal_features(
        cache_path,
        reranker_path=reranker_path,
        retrieval_recipe=retrieval_recipe,
        seed=seed,
        known_fraction=known_fraction,
        max_queries_per_id=max_queries_per_id,
        ann_topk=ann_topk,
        reranker_device=reranker_device,
    )
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed).fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    best = None
    for t in np.linspace(0.01, 0.99, 197):
        m = refusal_metrics(y, probs, float(t))
        key = (m["F1"], m["TNR"])
        if best is None or key > (best[0], best[1]):
            best = (m["F1"], m["TNR"], float(t), m)

    spec = {
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "threshold": best[2],
        "features": ["top1", "margin12", "global_top1", "top1_minus_top5mean"],
        "calibration_metrics": best[3],
        "retrieval_recipe": retrieval_recipe or {},
        "protocol": {
            "seed": int(seed),
            "known_fraction": float(known_fraction),
            "max_queries_per_id": int(max_queries_per_id),
            "ann_topk": int(ann_topk),
            "known_ids": len(protocol["known_ids"]),
            "unknown_ids": len(protocol["unknown_ids"]),
            "queries": int(len(y)),
        },
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = rows.copy(); rows["calibrated_probability"] = probs
    rows.to_csv(out_json.with_name(out_json.stem + "_calibration_queries.csv"), index=False)
    return spec


def evaluate_refusal(
    cache_path: str | Path,
    spec_or_path: dict | str | Path,
    *,
    reranker_path: str | Path | None = None,
    retrieval_recipe: dict | None = None,
    seed: int = 4242,
    reranker_device: str = "cpu",
) -> tuple[dict, pd.DataFrame]:
    spec = spec_or_path if isinstance(spec_or_path, dict) else json.loads(Path(spec_or_path).read_text(encoding="utf-8"))
    pcfg = spec.get("protocol", {})
    X, y, rows, protocol = build_refusal_features(
        cache_path,
        reranker_path=reranker_path,
        retrieval_recipe=retrieval_recipe if retrieval_recipe is not None else spec.get("retrieval_recipe", {}),
        seed=seed,
        known_fraction=float(pcfg.get("known_fraction", 0.60)),
        max_queries_per_id=int(pcfg.get("max_queries_per_id", 2)),
        ann_topk=int(pcfg.get("ann_topk", 100)),
        reranker_device=reranker_device,
    )
    probs = refusal_probability(spec, X)
    metrics = refusal_metrics(y, probs, float(spec["threshold"]))
    metrics.update({
        "known_ids": int(len(protocol["known_ids"])),
        "unknown_ids": int(len(protocol["unknown_ids"])),
        "protocol_seed": int(seed),
    })
    rows = rows.copy()
    rows["match_probability"] = probs
    rows["accepted"] = (probs >= float(spec["threshold"])).astype(np.uint8)
    rows["correct_acceptance"] = (rows["accepted"].to_numpy() == y).astype(np.uint8)
    return metrics, rows


def refusal_probability(spec: dict, x: np.ndarray) -> np.ndarray:
    w = np.asarray(spec["coef"], np.float32)
    b = float(spec["intercept"])
    s = np.asarray(x, np.float32) @ w + b
    return 1 / (1 + np.exp(-s))
