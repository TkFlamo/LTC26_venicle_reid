#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile

import pandas as pd

from vehicle_fingerprint.external.importers import (
    discover_veri776_root,
    import_veri776,
    write_veri776_report,
)


def _extract_zip(archive: Path, target: Path, force: bool = False) -> Path:
    archive = archive.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"VeRi archive not found: {archive}")
    if archive.suffix.lower() != ".zip":
        raise ValueError("--veri-archive currently expects the .zip downloaded from Kaggle")
    if target.exists() and any(target.iterdir()):
        if not force:
            print(f"[extract] target already contains files, reusing: {target}")
            return target
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive} -> {target}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    return target


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare the external ReID source used by v0.4: VeRi-776. "
                    "The importer auto-detects the Kaggle mirror's nested layout."
    )
    p.add_argument("--root", default="data/external",
                   help="External-data root. VeRi is searched recursively below this path.")
    p.add_argument("--veri-root", default=None,
                   help="Optional explicit extracted Kaggle/VeRi directory. It may be a wrapper directory; image_train is auto-discovered.")
    p.add_argument("--veri-archive", default=None,
                   help="Optional Kaggle .zip. If supplied it is extracted to <root>/VeRi before import.")
    p.add_argument("--force-extract", action="store_true",
                   help="Delete and recreate <root>/VeRi when --veri-archive is supplied.")
    p.add_argument("--out-dir", default="data/processed/external")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.veri_archive:
        source = _extract_zip(Path(args.veri_archive), root / "VeRi", force=args.force_extract)
    elif args.veri_root:
        source = Path(args.veri_root).expanduser().resolve()
    elif (root / "VeRi").exists():
        candidate = root / "VeRi"
        try:
            discover_veri776_root(candidate)
            source = candidate
        except Exception:
            # An empty/stale VeRi directory should not hide a Kaggle wrapper extracted
            # directly somewhere else below --root.
            source = root
    else:
        # Useful when the Kaggle zip was unpacked directly under data/external with its own wrapper name.
        source = root

    resolved = discover_veri776_root(source)
    print(f"[VeRi-776] supplied/search root: {source}")
    print(f"[VeRi-776] resolved dataset root: {resolved}")

    csv_path = out / "veri776.csv"
    df = import_veri776(source, csv_path)
    report = write_veri776_report(df, out / "veri776_import_report.json")

    train = df[df["split"].astype(str) == "train"].copy()
    if train.empty:
        raise SystemExit("VeRi importer produced no train split")
    train.to_csv(out / "reid_train.csv", index=False)

    # Dedicated manifests for the vendored V5 trainer. V5 resolves image_id
    # relative to --images-dir, so keep image_train/image_query/image_test in image_id.
    resolved_root = Path(report["resolved_root"]).resolve()
    def v5_manifest(frame: pd.DataFrame) -> pd.DataFrame:
        x = frame.copy()
        rel = []
        for path in x["path"].astype(str):
            pp = Path(path).resolve()
            try:
                rel.append(pp.relative_to(resolved_root).as_posix())
            except ValueError as e:
                raise RuntimeError(f"VeRi image is outside resolved root: {pp} vs {resolved_root}") from e
        x["image_id"] = rel
        return x[["image_id", "vehicle_id", "camera_id"]].reset_index(drop=True)

    v5_train = v5_manifest(train)
    v5_train.to_csv(out / "veri_v5_train.csv", index=False)

    # Preserve VeRi's own query/gallery protocol for external pretraining checkpoint selection.
    veri_query = df[df["split"].astype(str) == "query"].copy()
    veri_gallery = df[df["split"].astype(str) == "gallery"].copy()
    if not veri_query.empty and not veri_gallery.empty:
        veri_query.to_csv(out / "veri_query.csv", index=False)
        veri_gallery.to_csv(out / "veri_gallery.csv", index=False)
        gtq = veri_query[["image_id","vehicle_id","camera_id"]].copy(); gtq["split"] = "query"
        gtg = veri_gallery[["image_id","vehicle_id","camera_id"]].copy(); gtg["split"] = "gallery"
        pd.concat([gtq,gtg],ignore_index=True).to_csv(out / "veri_ground_truth.csv", index=False)

        # Standard VeRi train/test identities are disjoint. Query+gallery are validation-only
        # for the external checkpoint selector and are never used for gradient updates.
        v5_val = v5_manifest(pd.concat([veri_query, veri_gallery], ignore_index=True))
        overlap = set(v5_train.vehicle_id.astype(str)) & set(v5_val.vehicle_id.astype(str))
        if overlap:
            raise RuntimeError(f"VeRi train/validation identities overlap: {len(overlap)}")
        v5_val.to_csv(out / "veri_v5_val.csv", index=False)
        meta = {
            "resolved_root": str(resolved_root),
            "train_rows": int(len(v5_train)),
            "train_ids": int(v5_train.vehicle_id.astype(str).nunique()),
            "val_rows": int(len(v5_val)),
            "val_ids": int(v5_val.vehicle_id.astype(str).nunique()),
            "train_val_identity_overlap": 0,
            "note": "VeRi query+gallery are external-validation-only; hackathon validation is untouched",
        }
        (out / "veri_v5_transfer_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n[VeRi-776] imported split statistics")
    summary = df.groupby("split").agg(
        images=("sample_id", "size"),
        vehicle_ids=("vehicle_id", "nunique"),
        cameras=("camera_id", "nunique"),
    )
    print(summary.to_string())

    # Soft audit only: Kaggle mirrors sometimes contain slightly different packaging.
    n_train = len(train)
    n_ids = train["vehicle_id"].nunique()
    n_cams = train["camera_id"].nunique()
    if n_train < 30000:
        print(f"[warn] only {n_train} VeRi train images were imported; a full standard copy is usually much larger. Check veri776_import_report.json.")
    if n_ids < 500:
        print(f"[warn] only {n_ids} training vehicle IDs were imported; verify that image_train was found correctly.")
    if n_cams < 10:
        print(f"[warn] only {n_cams} training cameras were parsed; verify filename format <vehicle>_c<camera>_....jpg.")

    print(f"\n[saved] full VeRi audit manifest: {csv_path}")
    print(f"[saved] external ReID train manifest (VeRi train only): {out / 'reid_train.csv'}")
    print(f"[saved] V5 transfer train manifest: {out / 'veri_v5_train.csv'}")
    print(f"[saved] import report: {out / 'veri776_import_report.json'}")
    if (out / "veri_query.csv").exists():
        print(f"[saved] VeRi official query/gallery: {out / 'veri_query.csv'} / {out / 'veri_gallery.csv'}")
        print(f"[saved] VeRi validation GT: {out / 'veri_ground_truth.csv'}")
        print(f"[saved] V5 transfer validation manifest: {out / 'veri_v5_val.csv'}")
        print(f"[saved] V5 transfer manifest audit: {out / 'veri_v5_transfer_manifest.json'}")
    print("[note] image_query/image_test are validation-only and are never used for gradient updates.")


if __name__ == "__main__":
    main()
