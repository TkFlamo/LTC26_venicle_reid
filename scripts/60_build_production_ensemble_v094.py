#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_MEMBERS = (
    ("base_full", 0.65),
    ("external_convnext_full", 0.35),
)


def _copy_tree(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=True)
    elif mode == "copy":
        shutil.copytree(src, dst, symlinks=False)
    elif mode == "hardlink":
        def cp(a, b):
            try:
                os.link(a, b)
            except OSError:
                shutil.copy2(a, b)
        shutil.copytree(src, dst, copy_function=cp, symlinks=False)
    else:
        raise ValueError(mode)


def _validate_single(path: Path, name: str) -> dict:
    mp = path / "deployment.json"
    if not mp.is_file():
        raise FileNotFoundError(f"{name}: missing {mp}")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if str(meta.get("mode", "single")).lower() != "single":
        raise ValueError(f"{name}: expected mode=single, got {meta.get('mode')!r}")
    ck = path / str(meta.get("checkpoint", "reid.pt"))
    if not ck.is_file():
        raise FileNotFoundError(f"{name}: checkpoint missing: {ck}")
    recipe = path / str(meta.get("retrieval_recipe", "retrieval_recipe.json"))
    if not recipe.is_file():
        raise FileNotFoundError(f"{name}: retrieval recipe missing: {recipe}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the fixed production v0.9.4 ensemble: base_full 0.65 + external_convnext_full 0.35"
    )
    ap.add_argument("--base-full", required=True, help="Path to base_full single deployment")
    ap.add_argument("--external-convnext-full", required=True, help="Path to external_convnext_full single deployment")
    ap.add_argument("--out", default="deploy/models_current")
    ap.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    ap.add_argument(
        "--external-origin", choices=["team_trained", "public_external", "unresolved"], default="unresolved",
        help="Organizer provenance classification for external_convnext_full",
    )
    ap.add_argument("--external-source-url", default=None)
    ap.add_argument("--external-source-version", default=None)
    ap.add_argument("--external-source-sha256", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    srcs = {
        "base_full": Path(a.base_full).expanduser().resolve(),
        "external_convnext_full": Path(a.external_convnext_full).expanduser().resolve(),
    }
    out = Path(a.out).expanduser()
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    if out.exists():
        if not a.force:
            raise SystemExit(f"Output exists: {out}; use --force")
        if out.is_symlink() or out.is_file():
            out.unlink()
        else:
            shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    members = []
    for name, weight in PRODUCTION_MEMBERS:
        src = srcs[name]
        _validate_single(src, name)
        dst = out / "members" / name
        _copy_tree(src, dst, a.copy_mode)
        members.append({
            "name": name,
            "path": str(Path("members") / name),
            "weight": weight,
            "source": str(src),
            "source_mode": "single",
        })

    if a.external_origin == "public_external":
        missing = [
            name for name, value in (
                ("--external-source-url", a.external_source_url),
                ("--external-source-version", a.external_source_version),
                ("--external-source-sha256", a.external_source_sha256),
            ) if not value
        ]
        if missing:
            raise SystemExit(f"public_external requires {', '.join(missing)}")

    payload = {
        "schema": "vehicle-reid-v094-production-score-ensemble-v1",
        "mode": "score_ensemble",
        "profile": "production_base_external_convnext",
        "fusion": "per_query_zscore",
        "members": members,
        "candidate_policy": {
            "mode": "inherit_member",
            "member": "base_full",
            "note": "Refusal calibration belongs to base_full; score-fusion refusal was not recalibrated.",
        },
        "embedding_policy": {
            "mode": "weighted_concat",
            "weights": {"base_full": 0.65, "external_convnext_full": 0.35},
            "note": "embeddings.npy is a real fixed-dimensional representation of both deployed feature models; final submission ranking additionally uses documented per-query score fusion/re-ranking.",
        },
        "provenance": {
            "base_full": {
                "origin": "team_trained",
                "note": "Target-task deployment built by the team; document public backbone/data sources in MODEL_PROVENANCE.md.",
            },
            "external_convnext_full": {
                "origin": a.external_origin,
                "url": a.external_source_url,
                "version": a.external_source_version,
                "sha256": a.external_source_sha256,
                "note": (
                    "If this checkpoint was trained by another member of the same team, use team_trained. "
                    "If it is a third-party public checkpoint, public URL/version/SHA256 are required by organizer Q&A."
                ),
            },
        },
        "selection": {
            "variant": "Z::base_full+external_convnext_full::0.650,0.350",
            "shared_validation_mAP@10_mean": 0.882672769708484,
            "source": "all_metrics_summary.csv",
        },
        "runtime": {
            "feature_runtime": "pytorch_cuda",
            "exact_ranking": True,
            "shared_preprocess": True,
            "weight_format": ".pt",
            "recommended_precision": "fp16",
            "official_throughput_batches": [1, 8, 16, 32],
            "target_fps": 100.0,
        },
    }
    (out / "deployment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[OK] production deployment: {out}")


if __name__ == "__main__":
    main()
