#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vehicle_fingerprint.data.crop import find_image

REQUIRED_INPUT_COLUMNS = ["image_id", "x", "y", "w", "h"]
CANDIDATE_COLUMNS = ["query_id", "gallery_id", "confidence"]


def validate_input_csv(path: Path, images: Path, label: str) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise ValueError(f"Missing {label} CSV: {path}")
    df = pd.read_csv(path, dtype={"image_id": str})
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    if df["image_id"].isna().any() or (df["image_id"].astype(str).str.len() == 0).any():
        raise ValueError(f"{path.name}: empty image_id")
    if df["image_id"].astype(str).duplicated().any():
        bad = df.loc[df["image_id"].astype(str).duplicated(False), "image_id"].astype(str).head(10).tolist()
        raise ValueError(f"{path.name}: image_id must be unique; examples={bad}")
    for c in ("x", "y", "w", "h"):
        df[c] = pd.to_numeric(df[c], errors="raise")
        if not np.isfinite(df[c].to_numpy(float)).all():
            raise ValueError(f"{path.name}: non-finite {c}")
    if (df["w"] <= 0).any() or (df["h"] <= 0).any():
        raise ValueError(f"{path.name}: w/h must be positive")
    missing_images = []
    for image_id in df["image_id"].astype(str):
        try:
            find_image(images, image_id)
        except FileNotFoundError:
            missing_images.append(image_id)
            if len(missing_images) >= 20:
                break
    if missing_images:
        raise ValueError(f"{label}: image files not found for IDs {missing_images}")
    return df, df["image_id"].astype(str).tolist()


def validate_submission(path: Path, qids: list[str], gids: list[str]) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing submission.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) != len(qids):
        raise ValueError(f"submission.csv: expected {len(qids)} rows, got {len(rows)}")
    expected_k = min(10, len(gids))
    gallery_set = set(gids)
    for i, (row, expected_qid) in enumerate(zip(rows, qids), start=1):
        if not row:
            raise ValueError(f"submission.csv row {i}: empty row")
        if row[0] == "query_id":
            raise ValueError("submission.csv must NOT contain a header")
        if str(row[0]) != str(expected_qid):
            raise ValueError(f"submission.csv row {i}: expected query_id={expected_qid}, got {row[0]}")
        preds = [str(x) for x in row[1:]]
        if len(preds) != expected_k:
            raise ValueError(f"submission.csv row {i}: expected {expected_k} gallery ids, got {len(preds)}")
        if len(set(preds)) != len(preds):
            raise ValueError(f"submission.csv row {i}: duplicate gallery IDs")
        bad = [x for x in preds if x not in gallery_set]
        if bad:
            raise ValueError(f"submission.csv row {i}: unknown gallery IDs {bad[:5]}")
    return {"rows": len(rows), "topk": expected_k, "header": False}


def validate_candidates(path: Path, qids: list[str], gids: list[str]) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing candidates.csv: {path}")
    df = pd.read_csv(path, dtype={"query_id": str, "gallery_id": str})
    if list(df.columns) != CANDIDATE_COLUMNS:
        raise ValueError(f"candidates.csv header/order must be exactly {CANDIDATE_COLUMNS}; got {list(df.columns)}")
    qset, gset = set(qids), set(gids)
    if len(df):
        if not set(df["query_id"].astype(str)).issubset(qset):
            bad = sorted(set(df["query_id"].astype(str)) - qset)[:10]
            raise ValueError(f"candidates.csv: unknown query IDs {bad}")
        if not set(df["gallery_id"].astype(str)).issubset(gset):
            bad = sorted(set(df["gallery_id"].astype(str)) - gset)[:10]
            raise ValueError(f"candidates.csv: unknown gallery IDs {bad}")
        conf = pd.to_numeric(df["confidence"], errors="raise").to_numpy(float)
        if not np.isfinite(conf).all():
            raise ValueError("candidates.csv: confidence must be finite")
    refused = len(set(qids) - set(df["query_id"].astype(str)))
    return {"rows": int(len(df)), "refused_queries": int(refused), "multiple_candidates_allowed": True}


def validate_embeddings(path: Path, nq: int, ng: int) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing embeddings.npy: {path}")
    x = np.load(path, allow_pickle=False)
    if x.ndim != 2:
        raise ValueError(f"embeddings.npy must be 2D, got shape={x.shape}")
    if x.shape[0] != nq + ng:
        raise ValueError(f"embeddings.npy: expected {nq + ng} rows (query then gallery), got {x.shape[0]}")
    if x.dtype != np.float32:
        raise ValueError(f"embeddings.npy dtype must be float32, got {x.dtype}")
    if not np.isfinite(x).all():
        raise ValueError("embeddings.npy contains NaN/Inf")
    norms = np.linalg.norm(x, axis=1)
    return {
        "shape": [int(x.shape[0]), int(x.shape[1])],
        "dtype": str(x.dtype),
        "norm_mean": float(norms.mean()) if len(norms) else math.nan,
        "row_order": "all query rows, then all gallery rows",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Falcon Tech hackathon input and organizer submission contract")
    ap.add_argument("--input-dir", required=True, help="Contains test_query.csv, test_gallery.csv, images/")
    ap.add_argument("--output-dir", default=None, help="Optional generated submission directory")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    inp = Path(a.input_dir).expanduser().resolve()
    images = inp / "images"
    if not images.is_dir():
        raise SystemExit(f"Missing images/: {images}")
    qdf, qids = validate_input_csv(inp / "test_query.csv", images, "query")
    gdf, gids = validate_input_csv(inp / "test_gallery.csv", images, "gallery")
    overlap = sorted(set(qids) & set(gids))
    report = {
        "schema": "vehicle-reid-v094-hackathon-io-validation-v1",
        "input": {
            "query_rows": len(qids),
            "gallery_rows": len(gids),
            "columns_required": REQUIRED_INPUT_COLUMNS,
            "query_gallery_image_id_overlap": len(overlap),
            "one_object_per_image_id": True,
            "camera_id_required_at_test": False,
        },
    }
    if a.output_dir:
        out = Path(a.output_dir).expanduser().resolve()
        report["output"] = {
            "submission": validate_submission(out / "submission.csv", qids, gids),
            "candidates": validate_candidates(out / "candidates.csv", qids, gids),
            "embeddings": validate_embeddings(out / "embeddings.npy", len(qids), len(gids)),
        }
    if a.json_out:
        jp = Path(a.json_out).expanduser()
        if not jp.is_absolute(): jp = (ROOT / jp).resolve()
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("[OK] organizer I/O contract validation passed")


if __name__ == "__main__":
    main()
