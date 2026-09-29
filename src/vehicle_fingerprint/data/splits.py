from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _identity_components(df: pd.DataFrame) -> dict[str, list[str]]:
    """Join identities that co-occur in the same source image.

    This makes the split both identity-disjoint and source-image-disjoint. If two different
    vehicles appear in one frame, putting them on opposite sides would leak the exact same
    background/camera frame into validation.
    """
    vids = sorted(df["vehicle_id"].astype(str).unique())
    uf = _UnionFind(vids)
    if "image_id" in df.columns:
        for _, g in df.groupby(df["image_id"].astype(str), sort=False):
            group_vids = g["vehicle_id"].astype(str).unique().tolist()
            if len(group_vids) > 1:
                a = group_vids[0]
                for b in group_vids[1:]:
                    uf.union(a, b)
    comps: dict[str, list[str]] = defaultdict(list)
    for v in vids:
        comps[uf.find(v)].append(v)
    return dict(comps)


def identity_disjoint_train_val_eval_split(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.10,
    eval_fraction: float = 0.10,
    seed: int = 42,
    min_eval_cameras: int = 2,
) -> pd.DataFrame:
    """Create a leakage-resistant target-domain split.

    Properties:
    * no ``vehicle_id`` overlap between train / val / eval;
    * no exact ``image_id`` overlap between splits, including frames with multiple vehicles;
    * validation/evaluation components are selected from identities with at least
      ``min_eval_cameras`` distinct cameras so cross-camera ReID is actually measurable;
    * one-camera identities are retained in train instead of diluting final metrics.

    ``val`` is intended for checkpoint selection and refusal calibration. ``eval`` is the final
    untouched holdout used to report the competition metrics.
    """
    if "vehicle_id" not in df.columns:
        raise ValueError("vehicle_id is required")
    if val_fraction < 0 or eval_fraction < 0 or val_fraction + eval_fraction >= 1:
        raise ValueError("Require 0 <= val_fraction, eval_fraction and val_fraction + eval_fraction < 1")

    out = df.copy()
    vids = out["vehicle_id"].astype(str)
    out["vehicle_id"] = vids
    if "camera_id" not in out.columns:
        raise ValueError("camera_id is required for a cross-camera validation split")
    out["camera_id"] = out["camera_id"].astype(str)

    id_stats = out.groupby("vehicle_id").agg(
        n_images=("vehicle_id", "size"),
        n_cameras=("camera_id", "nunique"),
    )
    eligible = set(id_stats.index[id_stats.n_cameras >= int(min_eval_cameras)].astype(str))
    if not eligible and (val_fraction > 0 or eval_fraction > 0):
        raise ValueError("No identities have enough distinct cameras for cross-camera validation")

    comps = _identity_components(out)
    comp_records = []
    for root, members in comps.items():
        e = [v for v in members if v in eligible]
        comp_records.append({
            "root": root,
            "members": members,
            "eligible_ids": e,
            "eligible_count": len(e),
            "images": int(out[out.vehicle_id.isin(members)].shape[0]),
        })

    candidates = [c for c in comp_records if c["eligible_count"] > 0]
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    # Large connected components are assigned first; random tie order remains deterministic.
    candidates.sort(key=lambda c: (c["eligible_count"], c["images"]), reverse=True)

    n_eligible = len(eligible)
    target_val = int(round(n_eligible * val_fraction))
    target_eval = int(round(n_eligible * eval_fraction))
    if val_fraction > 0:
        target_val = max(1, target_val)
    if eval_fraction > 0:
        target_eval = max(1, target_eval)
    if target_val + target_eval > n_eligible:
        raise ValueError(
            f"Not enough cross-camera identities ({n_eligible}) for requested val/eval targets "
            f"({target_val}+{target_eval})"
        )

    assigned: dict[str, str] = {}
    counts = {"val": 0, "eval": 0}
    targets = {"val": target_val, "eval": target_eval}

    for c in candidates:
        deficits = {
            s: (targets[s] - counts[s]) / max(1, targets[s])
            for s in ("val", "eval")
            if targets[s] > 0 and counts[s] < targets[s]
        }
        if not deficits:
            break
        # Largest relative deficit; deterministic val-before-eval tie break after seeded shuffle.
        split = max(deficits, key=lambda s: (deficits[s], 1 if s == "val" else 0))
        for v in c["members"]:
            assigned[v] = split
        counts[split] += c["eligible_count"]

    out["split"] = [assigned.get(v, "train") for v in out["vehicle_id"].astype(str)]
    assert_split_disjoint(out)
    return out


def identity_disjoint_split(df: pd.DataFrame, val_fraction: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Backward-compatible train/val split by identity only."""
    if "vehicle_id" not in df.columns:
        raise ValueError("vehicle_id is required")
    ids = np.array(sorted(df["vehicle_id"].astype(str).unique()))
    if val_fraction <= 0:
        out = df.copy(); out["split"] = "train"; return out
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_fraction)))
    val_ids = set(ids[:n_val].tolist())
    out = df.copy()
    out["split"] = np.where(out["vehicle_id"].astype(str).isin(val_ids), "val", "train")
    assert_identity_disjoint(out)
    return out


def assert_identity_disjoint(df: pd.DataFrame) -> None:
    if "split" not in df.columns:
        return
    split_ids = {s: set(g["vehicle_id"].astype(str)) for s, g in df.groupby("split")}
    names = sorted(split_ids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = split_ids[a] & split_ids[b]
            if overlap:
                raise AssertionError(f"Identity leakage between {a} and {b}: {len(overlap)} ids")


def assert_split_disjoint(df: pd.DataFrame) -> None:
    assert_identity_disjoint(df)
    if "image_id" not in df.columns or "split" not in df.columns:
        return
    split_images = {s: set(g["image_id"].astype(str)) for s, g in df.groupby("split")}
    names = sorted(split_images)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = split_images[a] & split_images[b]
            if overlap:
                raise AssertionError(f"Source-image leakage between {a} and {b}: {len(overlap)} images")


def split_diagnostics(df: pd.DataFrame, min_eval_cameras: int = 2) -> dict:
    rows = {}
    for split, g in df.groupby("split", sort=True):
        per_id_cams = g.groupby(g["vehicle_id"].astype(str))["camera_id"].nunique()
        rows[str(split)] = {
            "samples": int(len(g)),
            "identities": int(g["vehicle_id"].astype(str).nunique()),
            "source_images": int(g["image_id"].astype(str).nunique()) if "image_id" in g else int(len(g)),
            "cameras": int(g["camera_id"].astype(str).nunique()),
            "cross_camera_identities": int((per_id_cams >= min_eval_cameras).sum()),
        }
    return rows
