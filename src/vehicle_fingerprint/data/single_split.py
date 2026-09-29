from __future__ import annotations

"""Deterministic one-split development protocol with immutable V5 shared validation.

This is the compute-efficient v0.9.4 alternative to outer K-fold training.  The cross-project
shared validation pool remains byte-for-byte fixed.  Only the remaining development identities
are split once into model train / inner validation.  The inner validation is further partitioned
into reranker-fit and retrieval-selection identities so the pair reranker is not evaluated on the
same identities used to fit it.
"""

from pathlib import Path
import json

import pandas as pd

from .kfold import (
    _choose_holdout,
    _component_records,
    _sha_lines,
    build_v5_shared_protocols,
    enrich_manifest,
    load_fixed_validation_spec,
)
from .official_protocol import build_official_protocol
from .splits import assert_split_disjoint


def _extract_fixed_shared_validation(raw: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = {"vehicle_id", "camera_id", "image_id"}
    miss = required - set(raw.columns)
    if miss:
        raise ValueError(f"Single-split build requires {sorted(miss)}")
    raw = raw.copy()
    for c in ("vehicle_id", "camera_id", "image_id"):
        raw[c] = raw[c].astype(str)
    if raw.image_id.duplicated().any():
        raise ValueError("Fixed shared validation expects unique image_id rows")

    ordered = spec["ordered_validation_rows"]
    val_ids_ordered = [str(r["image_id"]) for r in ordered]
    row_by_image = raw.set_index("image_id", drop=False)
    missing_images = [x for x in val_ids_ordered if x not in row_by_image.index]
    if missing_images:
        raise ValueError(
            f"Shared validation mismatch: {len(missing_images)} expected image_id values are missing; first={missing_images[:5]}"
        )
    shared_val = row_by_image.loc[val_ids_ordered].copy().reset_index(drop=True)
    for i, exp in enumerate(ordered):
        got = shared_val.iloc[i]
        if str(got.vehicle_id) != str(exp["vehicle_id"]) or str(got.camera_id) != str(exp["camera_id"]):
            raise AssertionError(
                f"Shared validation metadata mismatch for image_id={exp['image_id']}: "
                f"expected vehicle/camera={exp['vehicle_id']}/{exp['camera_id']} got={got.vehicle_id}/{got.camera_id}"
            )

    shared_images = set(val_ids_ordered)
    dev = raw[~raw.image_id.isin(shared_images)].copy().reset_index(drop=True)
    exp_counts = spec.get("dataset_expectations", {})
    checks = {
        "total_rows": len(raw),
        "total_vehicle_ids": raw.vehicle_id.nunique(),
        "development_rows": len(dev),
        "development_vehicle_ids": dev.vehicle_id.nunique(),
        "shared_validation_rows": len(shared_val),
        "shared_validation_vehicle_ids": shared_val.vehicle_id.nunique(),
    }
    for k, got in checks.items():
        if k in exp_counts and int(got) != int(exp_counts[k]):
            raise AssertionError(f"Shared validation dataset mismatch: {k} expected={exp_counts[k]} got={got}")

    val_vids = set(shared_val.vehicle_id.astype(str))
    dev_vids = set(dev.vehicle_id.astype(str))
    if val_vids & dev_vids:
        raise AssertionError(f"Identity leakage into shared validation: {len(val_vids & dev_vids)} vehicle IDs")

    actual_fp = {
        "val_vehicle_ids_sha256": _sha_lines(sorted(val_vids)),
        "val_image_ids_sorted_sha256": _sha_lines(sorted(shared_val.image_id.astype(str).tolist())),
        "val_image_ids_ordered_sha256": _sha_lines(shared_val.image_id.astype(str).tolist()),
        "dev_vehicle_ids_sha256": _sha_lines(sorted(dev_vids)),
        "dev_image_ids_sorted_sha256": _sha_lines(sorted(dev.image_id.astype(str).tolist())),
    }
    for k, got in actual_fp.items():
        if k in spec.get("fingerprints", {}) and got != spec["fingerprints"][k]:
            raise AssertionError(f"Shared validation fingerprint mismatch for {k}")
    return dev, shared_val, actual_fp


def _split_components(df: pd.DataFrame, *, fraction: float, seed: int, min_eval_cameras: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    recs = _component_records(df, min_eval_cameras)
    chosen, remaining = _choose_holdout(recs, fraction, seed)
    chosen_roots = {r["root"] for r in chosen}
    root_for_id = {}
    for r in recs:
        for vid in r["members"]:
            root_for_id[str(vid)] = r["root"]
    mask = df.vehicle_id.astype(str).map(lambda x: root_for_id[x] in chosen_roots)
    held = df[mask].copy().reset_index(drop=True)
    kept = df[~mask].copy().reset_index(drop=True)
    return kept, held


def build_shared_validation_single_split_manifests(
    manifest: str | Path | pd.DataFrame,
    out_dir: str | Path,
    *,
    validation_spec: str | Path,
    inner_val_fraction: float = 0.25,
    reranker_fit_fraction_of_val: float = 0.50,
    seed: int = 42,
    min_eval_cameras: int = 2,
    open_set_fraction: float = 0.20,
    max_queries_per_id: int = 2,
    compute_brightness: bool = True,
) -> dict:
    raw = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
    spec = load_fixed_validation_spec(validation_spec)
    dev, shared_val, actual_fp = _extract_fixed_shared_validation(raw, spec)

    dev = enrich_manifest(dev, compute_brightness=compute_brightness)
    shared_val = enrich_manifest(shared_val, compute_brightness=compute_brightness)
    train, val = _split_components(
        dev,
        fraction=float(inner_val_fraction),
        seed=int(seed) + 17,
        min_eval_cameras=int(min_eval_cameras),
    )
    train["split"] = "train"
    val["split"] = "val"
    assert_split_disjoint(pd.concat([train, val], ignore_index=True))

    # A second identity-disjoint cut inside the unseen inner validation is used only for
    # reranker fitting versus retrieval/threshold selection.  Neither subset is used for gradients
    # of the backbone/part model.
    reranker_selection_pool = val.drop(columns=["split"], errors="ignore").copy()
    selection_val, reranker_fit = _split_components(
        reranker_selection_pool,
        fraction=float(reranker_fit_fraction_of_val),
        seed=int(seed) + 2117,
        min_eval_cameras=int(min_eval_cameras),
    )
    reranker_fit["split"] = "val"
    selection_val["split"] = "val"
    assert_split_disjoint(pd.concat([reranker_fit.assign(split="train"), selection_val.assign(split="val")], ignore_index=True))

    out = Path(out_dir)
    inner = out / "inner"
    inner.mkdir(parents=True, exist_ok=True)
    dev.to_csv(out / "dev.csv", index=False)
    shared_val.to_csv(out / "shared_val.csv", index=False)
    pd.concat([
        train.assign(dev_role="train"),
        val.assign(dev_role="inner_val"),
        shared_val.assign(split="shared_validation", dev_role="shared_validation"),
    ], ignore_index=True, sort=False).to_csv(out / "manifest_enriched.csv", index=False)
    train.to_csv(inner / "train.csv", index=False)
    val.to_csv(inner / "val.csv", index=False)
    reranker_fit.to_csv(inner / "reranker_fit.csv", index=False)
    selection_val.to_csv(inner / "selection_val.csv", index=False)

    inner_proto = build_official_protocol(
        val,
        inner / "official_val",
        seed=int(seed) + 1000,
        open_set_fraction=float(open_set_fraction),
        max_queries_per_id=int(max_queries_per_id),
    )
    reranker_proto = build_official_protocol(
        reranker_fit,
        inner / "reranker_official",
        seed=int(seed) + 2000,
        open_set_fraction=float(open_set_fraction),
        max_queries_per_id=int(max_queries_per_id),
    )
    selection_proto = build_official_protocol(
        selection_val,
        inner / "selection_official",
        seed=int(seed) + 3000,
        open_set_fraction=float(open_set_fraction),
        max_queries_per_id=int(max_queries_per_id),
    )
    shared_proto = build_v5_shared_protocols(shared_val, out / "shared_validation", spec)

    summary = {
        "mode": "fixed_shared_validation_plus_single_development_split",
        "validation_spec": str(validation_spec),
        "seed": int(seed),
        "inner_val_fraction": float(inner_val_fraction),
        "reranker_fit_fraction_of_val": float(reranker_fit_fraction_of_val),
        "rows": int(len(raw)),
        "identities": int(raw.vehicle_id.nunique()),
        "dev_rows": int(len(dev)),
        "dev_ids": int(dev.vehicle_id.nunique()),
        "train_rows": int(len(train)),
        "train_ids": int(train.vehicle_id.nunique()),
        "inner_val_rows": int(len(val)),
        "inner_val_ids": int(val.vehicle_id.nunique()),
        "reranker_fit_rows": int(len(reranker_fit)),
        "reranker_fit_ids": int(reranker_fit.vehicle_id.nunique()),
        "selection_val_rows": int(len(selection_val)),
        "selection_val_ids": int(selection_val.vehicle_id.nunique()),
        "shared_validation_rows": int(len(shared_val)),
        "shared_validation_ids": int(shared_val.vehicle_id.nunique()),
        "inner_official_protocol": inner_proto,
        "reranker_official_protocol": reranker_proto,
        "selection_official_protocol": selection_proto,
        "shared_validation_protocol": shared_proto,
        "shared_validation_fingerprints": actual_fp,
        "selection_policy": "all architecture/epoch/part/reranker choices use only development data; cross-project shared validation is reporting-only",
    }
    (out / "single_split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
