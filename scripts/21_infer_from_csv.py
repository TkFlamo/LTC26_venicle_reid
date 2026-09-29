#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

# Always import the package from THIS dist.  This must happen before importing
# vehicle_fingerprint so an inherited PYTHONPATH cannot silently select another
# checkout on Windows or Linux.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vehicle_fingerprint.data.prepare import prepare_hackathon_dataset
from vehicle_fingerprint.features import extract_feature_cache
from vehicle_fingerprint.retrieval import run_retrieval


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end base_full inference from organizer CSV + image folders"
    )
    p.add_argument("--query-csv", required=True)
    p.add_argument("--query-images", required=True)
    p.add_argument("--gallery-csv")
    p.add_argument("--gallery-images")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reranker")
    p.add_argument("--refusal")
    p.add_argument("--recipe")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--precision", default="bf16")
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--image-size", type=int, nargs=2, metavar=("H", "W"), default=None,
        help="Optional override. Omit to use checkpoint preprocessing, recommended.",
    )
    p.add_argument("--query-id-column", default=None)
    p.add_argument("--gallery-id-column", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)

    out = Path(a.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    qdir = work / "query"
    prepare_hackathon_dataset(
        a.query_csv, a.query_images, qdir,
        pad=0.03, val_fraction=0.0, eval_fraction=0.0,
        materialize_crops=False,
    )
    qmanifest = qdir / "test.csv" if (qdir / "test.csv").exists() else qdir / "manifest.csv"
    qcache = work / "query_features.npz"
    extract_feature_cache(
        qmanifest, a.checkpoint, qcache,
        device=a.device, precision=a.precision,
        batch_size=a.batch, workers=a.workers,
        image_size=a.image_size,
    )

    if a.gallery_csv:
        if not a.gallery_images:
            raise SystemExit("--gallery-images is required with --gallery-csv")
        gdir = work / "gallery"
        prepare_hackathon_dataset(
            a.gallery_csv, a.gallery_images, gdir,
            pad=0.03, val_fraction=0.0, eval_fraction=0.0,
            materialize_crops=False,
        )
        gmanifest = gdir / "test.csv" if (gdir / "test.csv").exists() else gdir / "manifest.csv"
        gcache = work / "gallery_features.npz"
        extract_feature_cache(
            gmanifest, a.checkpoint, gcache,
            device=a.device, precision=a.precision,
            batch_size=a.batch, workers=a.workers,
            image_size=a.image_size,
        )
    else:
        gcache = qcache

    result = run_retrieval(
        qcache, gcache, out,
        reranker_path=a.reranker,
        refusal_json=a.refusal,
        retrieval_recipe_json=a.recipe,
        query_id_column=a.query_id_column,
        gallery_id_column=a.gallery_id_column,
    )
    print(result)


if __name__ == "__main__":
    # Required for Windows DataLoader(num_workers>0), which uses spawn rather
    # than fork.  Child processes import this module but do not execute main().
    mp.freeze_support()
    main()
