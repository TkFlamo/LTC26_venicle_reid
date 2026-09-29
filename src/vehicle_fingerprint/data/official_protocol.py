from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _camera_diverse_rows(g: pd.DataFrame, rng: np.random.Generator, max_queries_per_id: int) -> list[int]:
    """Pick query rows while reserving one camera as a guaranteed cross-camera gallery anchor."""
    cams = sorted(g["camera_id"].astype(str).unique().tolist())
    if len(cams) < 2:
        return []
    anchor_cam = str(rng.choice(cams))
    query_cams = [c for c in cams if c != anchor_cam]
    rng.shuffle(query_cams)
    chosen: list[int] = []
    for cam in query_cams:
        inds = g.index[g["camera_id"].astype(str) == cam].to_numpy(dtype=np.int64)
        if len(inds):
            chosen.append(int(rng.choice(inds)))
            if len(chosen) >= max(1, int(max_queries_per_id)):
                break
    return chosen


def _open_query_rows(g: pd.DataFrame, rng: np.random.Generator, max_queries_per_id: int) -> list[int]:
    """Pick camera-diverse open-set queries. No row of this identity will enter gallery."""
    chosen: list[int] = []
    cams = sorted(g["camera_id"].astype(str).unique().tolist())
    rng.shuffle(cams)
    for cam in cams:
        inds = g.index[g["camera_id"].astype(str) == cam].to_numpy(dtype=np.int64)
        if len(inds):
            chosen.append(int(rng.choice(inds)))
            if len(chosen) >= max(1, int(max_queries_per_id)):
                break
    if not chosen and len(g):
        chosen = [int(rng.choice(g.index.to_numpy(dtype=np.int64)))]
    return chosen


def build_official_protocol(
    manifest: str | Path | pd.DataFrame,
    out_dir: str | Path,
    *,
    seed: int = 42,
    open_set_fraction: float = 0.20,
    max_queries_per_id: int = 2,
) -> dict:
    """Create a deterministic local query/gallery protocol consumed by organizers' evaluate.py.

    The input manifest must already be identity-disjoint from training. Known identities need at
    least two cameras. A fraction of identities is query-only so the same fixed split can also be
    used to calibrate official candidates.csv refusal metrics.
    """
    df = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
    required = {"image_id", "vehicle_id", "camera_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Official protocol requires columns: {sorted(missing)}")
    if not df["image_id"].astype(str).is_unique:
        dup = df.loc[df["image_id"].astype(str).duplicated(), "image_id"].astype(str).head().tolist()
        raise ValueError(f"image_id must be unique for the official evaluator; duplicates include {dup}")

    df = df.reset_index(drop=True).copy()
    df["image_id"] = df["image_id"].astype(str)
    df["vehicle_id"] = df["vehicle_id"].astype(str)
    df["camera_id"] = df["camera_id"].astype(str)
    ids = sorted(df["vehicle_id"].unique().tolist())
    if len(ids) < 3:
        raise ValueError("Need at least three identities for a useful official validation protocol")

    stats = df.groupby("vehicle_id")["camera_id"].nunique()
    known_eligible = set(stats.index[stats >= 2].astype(str).tolist())
    if not known_eligible:
        raise ValueError("No identities have >=2 cameras; official ranking cannot be measured")

    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(ids, dtype=object)
    rng.shuffle(shuffled)
    target_open = int(round(len(ids) * float(open_set_fraction)))
    if open_set_fraction > 0:
        target_open = max(1, target_open)
    target_open = min(target_open, max(0, len(ids) - 1))

    # Prefer one-camera identities as open-set because they cannot contribute to cross-camera mAP.
    one_cam = [x for x in shuffled.tolist() if x not in known_eligible]
    multi_cam = [x for x in shuffled.tolist() if x in known_eligible]
    open_ids = one_cam[:target_open]
    if len(open_ids) < target_open:
        open_ids += multi_cam[: target_open - len(open_ids)]
    open_ids = set(map(str, open_ids))
    known_ids = sorted(known_eligible - open_ids)
    if not known_ids:
        # Keep at least one cross-camera identity for ranking.
        rescue = next(iter(open_ids & known_eligible), None)
        if rescue is None:
            raise ValueError("Open-set fraction leaves no known cross-camera identity")
        open_ids.remove(rescue)
        known_ids = [rescue]

    query_idx: list[int] = []
    gallery_idx: list[int] = []
    for vid in known_ids:
        g = df[df["vehicle_id"] == vid]
        q = _camera_diverse_rows(g, rng, max_queries_per_id)
        if not q:
            continue
        query_idx.extend(q)
        qset = set(q)
        gallery_idx.extend([int(i) for i in g.index.tolist() if int(i) not in qset])

    for vid in sorted(open_ids):
        g = df[df["vehicle_id"] == vid]
        query_idx.extend(_open_query_rows(g, rng, max_queries_per_id))

    # Drop accidental duplicate rows while preserving deterministic order.
    query_idx = list(dict.fromkeys(query_idx))
    gallery_idx = list(dict.fromkeys(gallery_idx))
    qdf = df.loc[query_idx].reset_index(drop=True)
    gdf = df.loc[gallery_idx].reset_index(drop=True)
    if qdf.empty or gdf.empty:
        raise ValueError("Official protocol produced an empty query or gallery")

    gt_q = qdf[["image_id", "vehicle_id", "camera_id"]].copy(); gt_q["split"] = "query"
    gt_g = gdf[["image_id", "vehicle_id", "camera_id"]].copy(); gt_g["split"] = "gallery"
    gt = pd.concat([gt_q, gt_g], ignore_index=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    qdf.to_csv(out / "query.csv", index=False)
    gdf.to_csv(out / "gallery.csv", index=False)
    gt.to_csv(out / "ground_truth.csv", index=False)

    gal_vids = set(gdf["vehicle_id"].astype(str))
    open_queries = int((~qdf["vehicle_id"].astype(str).isin(gal_vids)).sum())
    known_queries = int(len(qdf) - open_queries)
    report = {
        "source_rows": int(len(df)),
        "source_identities": int(df["vehicle_id"].nunique()),
        "query_rows": int(len(qdf)),
        "gallery_rows": int(len(gdf)),
        "known_query_rows": known_queries,
        "open_set_query_rows": open_queries,
        "gallery_identities": int(gdf["vehicle_id"].nunique()),
        "seed": int(seed),
        "open_set_fraction_requested": float(open_set_fraction),
        "max_queries_per_id": int(max_queries_per_id),
        "open_set_identity_count": int(len(open_ids)),
        "known_identity_count": int(len(set(known_ids))),
        "note": "Use official/evaluate.py unchanged for all scoring. Query/gallery are fixed once and reused by every model.",
    }
    (out / "protocol.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
