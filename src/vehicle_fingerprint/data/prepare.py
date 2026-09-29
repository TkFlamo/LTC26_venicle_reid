from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .crop import crop_bbox, find_image, target_bbox_in_crop
from .splits import (
    identity_disjoint_train_val_eval_split,
    assert_split_disjoint,
    split_diagnostics,
)


def _sample_id(image_id: str, row_idx: int, vehicle_id: str | int | None) -> str:
    raw = f"{image_id}|{row_idx}|{vehicle_id}".encode()
    return hashlib.sha1(raw).hexdigest()[:20]


def prepare_hackathon_dataset(
    csv_path: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    *,
    pad: float = 0.08,
    val_fraction: float = 0.10,
    eval_fraction: float = 0.10,
    seed: int = 42,
    min_eval_cameras: int = 2,
    materialize_crops: bool = True,
    jpeg_quality: int = 95,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {"image_id", "x", "y", "w", "h"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    has_ids = "vehicle_id" in df.columns and df["vehicle_id"].notna().any()
    if has_ids:
        if "camera_id" not in df.columns:
            raise ValueError(
                "camera_id is required to build the competition-style cross-camera validation split"
            )
        df = identity_disjoint_train_val_eval_split(
            df,
            val_fraction=val_fraction,
            eval_fraction=eval_fraction,
            seed=seed,
            min_eval_cameras=min_eval_cameras,
        )
        assert_split_disjoint(df)
    else:
        df["split"] = "test"
        df["vehicle_id"] = ""
        if "camera_id" not in df.columns:
            df["camera_id"] = -1

    records = []
    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing crops"):
        image_id = str(row.image_id)
        vid = row.vehicle_id
        sid = _sample_id(image_id, int(row_idx), vid)
        crop_path = crops_dir / f"{sid}.jpg"
        src = find_image(images_dir, image_id)
        with Image.open(src) as im:
            source_size = im.size
            geom = target_bbox_in_crop(source_size, row.x, row.y, row.w, row.h, pad=pad)
            if materialize_crops and not crop_path.exists():
                im = im.convert("RGB")
                crop = crop_bbox(im, row.x, row.y, row.w, row.h, pad=pad)
                crop.save(crop_path, quality=jpeg_quality, subsampling=0)
        records.append({
            "sample_id": sid,
            "path": str(crop_path.resolve()) if materialize_crops else str(src.resolve()),
            "source_path": str(src.resolve()),
            "image_id": image_id,
            "source_row": int(row_idx),
            "vehicle_id": str(vid) if has_ids else "",
            "camera_id": str(row.camera_id),
            "dataset": "hackathon",
            "split": str(row.split),
            "x": float(row.x), "y": float(row.y), "w": float(row.w), "h": float(row.h),
            "bbox_pad": float(pad),
            **geom,
        })
    out = pd.DataFrame(records)
    out.to_csv(out_dir / "manifest.csv", index=False)
    for split, g in out.groupby("split"):
        g.to_csv(out_dir / f"{split}.csv", index=False)

    if has_ids:
        report = split_diagnostics(out, min_eval_cameras=min_eval_cameras)
        (out_dir / "split_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return out
