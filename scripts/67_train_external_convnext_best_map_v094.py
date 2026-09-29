#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party" / "external_convnext_v5_training"
PROFILE = "small_camera_sampler_strongerase"
REFERENCE_MAP10 = 0.8214526158811873
REFERENCE_BEST_EPOCH = 25


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def _check_dataset(data: Path) -> dict:
    train_csv = data / "train.csv"
    images = data / "images"
    if not train_csv.is_file():
        raise FileNotFoundError(f"Missing training manifest: {train_csv}")
    if not images.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images}")

    # Avoid importing pandas here so --check-only remains a very light preflight.
    header = train_csv.open("r", encoding="utf-8-sig").readline().strip().split(",")
    required = {"image_id", "vehicle_id", "camera_id", "x", "y", "w", "h"}
    missing = sorted(required - set(header))
    if missing:
        raise RuntimeError(f"train.csv missing required columns: {missing}; header={header}")
    return {"train_csv": str(train_csv), "images": str(images), "columns": header}


def _verify_vendor() -> dict:
    required = [
        "vehicle_reid_v5.py",
        "train_best_v5.py",
        "run_next_experiments_v7.py",
        "requirements_v5.txt",
        "SOURCE_SHA256.txt",
    ]
    missing = [x for x in required if not (VENDOR / x).is_file()]
    if missing:
        raise FileNotFoundError(f"Vendored external trainer incomplete: {missing}")
    return {x: sha256(VENDOR / x) for x in required if x != "SOURCE_SHA256.txt"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproduce the team developer ConvNeXt-Small raw checkpoint (convnext_best_map.pt) from hackathon train data"
    )
    ap.add_argument("--data", required=True, help="Dataset root containing train.csv and images/")
    ap.add_argument("--out", default="runs/external_convnext_v5/E3e_camera_strongerase")
    ap.add_argument("--final-checkpoint", default="weights/external/convnext_best_map.pt")
    ap.add_argument("--device", default="", help="Optional torch device, e.g. cuda:0")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="Overwrite final-checkpoint if it exists")
    a = ap.parse_args()

    data = Path(a.data).expanduser().resolve()
    out = _resolve(a.out)
    final_ckpt = _resolve(a.final_checkpoint)
    dataset_info = _check_dataset(data)
    vendor_hashes = _verify_vendor()

    command = [
        sys.executable,
        "train_best_v5.py",
        "--profile", PROFILE,
        "--data", str(data),
        "--out", str(out),
        "--precision", a.precision,
        "--workers", str(a.workers),
        "--eval-batch-size", str(a.eval_batch_size),
        # Make the selected run's seeds explicit even though train_best_v5 already fixes split/fold.
        "--seed", "42",
        "--split-seed", "42",
        "--fold-seed", "59",
    ]
    if a.device:
        command += ["--device", a.device]

    preflight = {
        "schema": "vehicle-reid-v094-external-convnext-raw-training-v1",
        "vendor": str(VENDOR),
        "profile": PROFILE,
        "dataset": dataset_info,
        "out": str(out),
        "selected_checkpoint": str(out / "best_map.pt"),
        "final_checkpoint": str(final_ckpt),
        "reference_best_epoch": REFERENCE_BEST_EPOCH,
        "reference_best_official_mAP@10": REFERENCE_MAP10,
        "source_hashes": vendor_hashes,
        "command": command,
        "note": "Reference metric is provenance only; exact floating-point replay can differ across GPU/CUDA/PyTorch versions.",
    }
    if a.check_only:
        print(json.dumps({**preflight, "status": "OK"}, ensure_ascii=False, indent=2))
        return

    selected = out / "best_map.pt"
    if final_ckpt.exists() and not a.force:
        raise SystemExit(f"Final checkpoint exists: {final_ckpt}; use --force")
    out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    gpu = env.get("GPU", "").strip()
    if gpu and not a.device:
        # Preserve the project's conventional GPU=<physical id> launcher style.
        env.setdefault("CUDA_VISIBLE_DEVICES", gpu)

    print("=" * 80)
    print("Reproducing external developer ConvNeXt checkpoint")
    print("profile:", PROFILE)
    print("vendor :", VENDOR)
    print("data   :", data)
    print("out    :", out)
    print("cmd    :", " ".join(map(str, command)))
    print("=" * 80, flush=True)
    subprocess.run(command, cwd=VENDOR, env=env, check=True)

    if not selected.is_file():
        raise FileNotFoundError(f"Trainer completed but selected checkpoint is missing: {selected}")
    final_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, final_ckpt)

    manifest = {
        **preflight,
        "status": "OK",
        "selected_checkpoint_sha256": sha256(selected),
        "final_checkpoint_sha256": sha256(final_ckpt),
        "selected_checkpoint_size_bytes": selected.stat().st_size,
    }
    manifest_path = final_ckpt.with_suffix(final_ckpt.suffix + ".training.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[OK] raw developer checkpoint reproduced: {final_ckpt}")


if __name__ == "__main__":
    main()
