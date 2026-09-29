#!/usr/bin/env python3
from __future__ import annotations

"""Inference for a deployed two-model score-equivalent v0.9.4 ensemble."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.data.prepare import prepare_hackathon_dataset
from vehicle_fingerprint.features import extract_feature_cache, ensemble_feature_caches
from vehicle_fingerprint.retrieval import run_retrieval


def main():
    p = argparse.ArgumentParser(description="End-to-end inference for a two-checkpoint v0.9.4 ensemble")
    p.add_argument("--query-csv", required=True)
    p.add_argument("--query-images", required=True)
    p.add_argument("--gallery-csv")
    p.add_argument("--gallery-images")
    p.add_argument("--checkpoint-a", required=True, help="First checkpoint; normally ConvNeXt")
    p.add_argument("--checkpoint-b", required=True, help="Second checkpoint; normally ViT")
    p.add_argument("--weight-a", type=float, required=True, help="Score weight of checkpoint A")
    p.add_argument("--reranker")
    p.add_argument("--refusal")
    p.add_argument("--recipe")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--precision", default="bf16")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--image-size", type=int, nargs=2, metavar=("H","W"), default=None)
    p.add_argument("--query-id-column", default=None)
    p.add_argument("--gallery-id-column", default=None)
    a = p.parse_args()

    if not 0.0 <= a.weight_a <= 1.0:
        raise SystemExit("--weight-a must be in [0,1]")

    out = Path(a.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)

    qdir = work / "query"
    prepare_hackathon_dataset(
        a.query_csv, a.query_images, qdir,
        pad=.03, val_fraction=.0, eval_fraction=.0, materialize_crops=False
    )
    qmanifest = qdir / "test.csv" if (qdir / "test.csv").exists() else qdir / "manifest.csv"

    qa = work / "query_a.npz"
    qb = work / "query_b.npz"
    qe = work / "query_ensemble.npz"
    extract_feature_cache(qmanifest, a.checkpoint_a, qa, device=a.device, precision=a.precision,
                          batch_size=a.batch, workers=a.workers, image_size=a.image_size)
    extract_feature_cache(qmanifest, a.checkpoint_b, qb, device=a.device, precision=a.precision,
                          batch_size=a.batch, workers=a.workers, image_size=a.image_size)
    ensemble_feature_caches(qa, qb, qe, weight_a=a.weight_a)

    if a.gallery_csv:
        if not a.gallery_images:
            raise SystemExit("--gallery-images is required with --gallery-csv")
        gdir = work / "gallery"
        prepare_hackathon_dataset(
            a.gallery_csv, a.gallery_images, gdir,
            pad=.03, val_fraction=.0, eval_fraction=.0, materialize_crops=False
        )
        gmanifest = gdir / "test.csv" if (gdir / "test.csv").exists() else gdir / "manifest.csv"
        ga = work / "gallery_a.npz"
        gb = work / "gallery_b.npz"
        ge = work / "gallery_ensemble.npz"
        extract_feature_cache(gmanifest, a.checkpoint_a, ga, device=a.device, precision=a.precision,
                              batch_size=a.batch, workers=a.workers, image_size=a.image_size)
        extract_feature_cache(gmanifest, a.checkpoint_b, gb, device=a.device, precision=a.precision,
                              batch_size=a.batch, workers=a.workers, image_size=a.image_size)
        ensemble_feature_caches(ga, gb, ge, weight_a=a.weight_a)
    else:
        ge = qe

    result = run_retrieval(
        qe, ge, out,
        reranker_path=a.reranker,
        refusal_json=a.refusal,
        retrieval_recipe_json=a.recipe,
        query_id_column=a.query_id_column,
        gallery_id_column=a.gallery_id_column,
    )
    print(result)


if __name__ == "__main__":
    main()
