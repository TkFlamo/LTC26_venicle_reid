from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve


def _valid_gallery_mask(
    i: int,
    vehicle_ids: np.ndarray,
    camera_ids: np.ndarray,
    *,
    same_camera_policy: Literal["same_identity", "all"] = "same_identity",
) -> np.ndarray:
    n = len(vehicle_ids)
    valid = np.ones(n, dtype=bool)
    valid[i] = False
    if same_camera_policy == "same_identity":
        # Standard vehicle/person ReID protocol: remove junk positives from the same identity
        # and same camera, while same-camera different identities remain valid negatives.
        valid &= ~((vehicle_ids == vehicle_ids[i]) & (camera_ids == camera_ids[i]))
    elif same_camera_policy == "all":
        # Non-official stricter diagnostic that excludes every same-camera candidate.
        valid &= camera_ids != camera_ids[i]
    else:
        raise ValueError(f"Unknown same_camera_policy={same_camera_policy!r}")
    return valid


def _ranking_stats(relevance: np.ndarray) -> tuple[float, float, int]:
    rel = relevance.astype(np.float32)
    n_pos = int(rel.sum())
    if n_pos <= 0:
        return 0.0, 0.0, 0
    ranks = np.arange(1, len(rel) + 1, dtype=np.float32)
    csum = np.cumsum(rel)
    precision = csum / ranks
    ap = float((precision * rel).sum() / n_pos)
    last_positive_rank = int(np.flatnonzero(rel > 0)[-1] + 1)
    inp = float(n_pos / max(1, last_positive_rank))
    return ap, inp, last_positive_rank


def cross_camera_retrieval_metrics(
    embeddings: np.ndarray,
    vehicle_ids: np.ndarray,
    camera_ids: np.ndarray,
    *,
    topk: tuple[int, ...] = (1, 5),
    same_camera_policy: Literal["same_identity", "all"] = "same_identity",
) -> dict[str, float]:
    """All-vs-all competition-style cross-camera retrieval metrics.

    A query contributes only when at least one positive from another camera exists.  The official organizer rule removes only same-identity + same-camera junk. ``all`` is retained only as a stricter diagnostic.
    """
    x = embeddings.astype(np.float32, copy=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    vehicle_ids = np.asarray(vehicle_ids).astype(str)
    camera_ids = np.asarray(camera_ids).astype(str)
    sims = x @ x.T
    n = len(x)
    aps: list[float] = []
    inps: list[float] = []
    hits = {k: [] for k in topk}
    skipped = 0

    for i in range(n):
        valid = _valid_gallery_mask(i, vehicle_ids, camera_ids, same_camera_policy=same_camera_policy)
        positives = valid & (vehicle_ids == vehicle_ids[i]) & (camera_ids != camera_ids[i])
        if not positives.any():
            skipped += 1
            continue
        inds = np.flatnonzero(valid)
        order = inds[np.argsort(-sims[i, inds], kind="stable")]
        rel = positives[order]
        ap, inp, _ = _ranking_stats(rel)
        aps.append(ap)
        inps.append(inp)
        for k in topk:
            hits[k].append(float(rel[:k].any()))

    out: dict[str, float] = {
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "mINP": float(np.mean(inps)) if inps else 0.0,
        "queries": int(len(aps)),
        "skipped_no_cross_camera_positive": int(skipped),
    }
    for k in topk:
        out[f"Rank-{k}"] = float(np.mean(hits[k])) if hits[k] else 0.0
    return out


def refusal_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int64)
    probs = np.asarray(probs).astype(np.float64)
    pred = probs >= threshold
    y = y_true.astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-12, p + r)
    tnr = tn / max(1, tn + fp)

    if len(np.unique(y_true)) >= 2:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, probs)
        # sklearn returns recall in decreasing order, so reverse before trapezoidal integration.
        pr_auc = float(auc(recall_curve[::-1], precision_curve[::-1]))
        ap = float(average_precision_score(y_true, probs))
    else:
        pr_auc = 0.0
        ap = 0.0

    return {
        "precision": float(p),
        "recall": float(r),
        "F1": float(f1),
        "TNR": float(tnr),
        "PR-AUC": pr_auc,
        "AveragePrecision": ap,
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "queries": int(len(y_true)),
    }
