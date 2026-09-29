#!/usr/bin/env python3
from __future__ import annotations

"""
Evaluate the user's two v0.9.4 global/base models (ViT + ConvNeXt) and their
fixed-weight score-equivalent ensemble on the immutable V5 shared validation.

IMPORTANT:
- evaluation only; no training/refit/calibration;
- ensemble weight is fixed BEFORE shared-validation evaluation;
- default ConvNeXt weight 0.45, ViT weight 0.55;
- official ranking metrics only (mAP@10 / Rank-1 / Rank-5 / mAP_full / mINP).
  Candidate/refusal metrics are intentionally not reported because a global ensemble
  requires its own refusal calibration on development data.

Canonical ensemble:
    z = concat(sqrt(w_vit) * L2(vit), sqrt(w_cn) * L2(convnext))
so cosine(z_q, z_g) == w_vit*cos(vit_q,vit_g) + w_cn*cos(cn_q,cn_g).

Example:
GPU=0 python scripts/48_eval_global_ensemble_shared_v5.py \
  --vit-checkpoint /path/to/vit/project_best.pt \
  --convnext-checkpoint /path/to/convnext/project_best.pt \
  --convnext-weight 0.45 \
  --out artifacts/shared_validation_global_ensemble \
  --device 0 --precision bf16 --batch 24
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.features import extract_feature_cache, load_cache
from vehicle_fingerprint.official_eval import official_metrics_from_embeddings


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _ensemble(vit: np.ndarray, cn: np.ndarray, cn_weight: float) -> np.ndarray:
    wcn = float(cn_weight)
    if not (0.0 <= wcn <= 1.0):
        raise ValueError(f"convnext_weight must be in [0,1], got {wcn}")
    wvit = 1.0 - wcn
    zv = _l2(vit)
    zc = _l2(cn)
    z = np.concatenate([
        math.sqrt(wvit) * zv,
        math.sqrt(wcn) * zc,
    ], axis=1)
    # Numerically normalize once more. In exact arithmetic norm is already one.
    return _l2(z)


def _pick_ids(cache: dict, manifest: Path) -> list[str]:
    if "meta_image_id" in cache:
        return cache["meta_image_id"].astype(str).tolist()
    df = pd.read_csv(manifest, dtype={"image_id": str})
    return df["image_id"].astype(str).tolist()


def _metric_row(model: str, fold: int, report: dict) -> dict:
    r = report.get("ranking", {})
    fr = report.get("full_ranking", {})
    return {
        "model": model,
        "fold": int(fold),
        "mAP@10": float(r.get("mAP@10", 0.0)),
        "Rank-1": float(r.get("Rank-1", 0.0)),
        "Rank-5": float(r.get("Rank-5", 0.0)),
        "mAP_full": float(fr.get("mAP_full", 0.0)),
        "mINP": float(fr.get("mINP", 0.0)),
        "n_scored": int(r.get("n_scored", 0)),
        "n_openset_excluded": int(r.get("n_openset_excluded", 0)),
    }


def _checkpoint_meta(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("model_cfg", {}) if isinstance(ck, dict) else {}
    bb = cfg.get("backbone", {}) if isinstance(cfg, dict) else {}
    prep = cfg.get("preprocess", {}) if isinstance(cfg, dict) else {}
    return {
        "path": str(path.resolve()),
        "profile": bb.get("profile"),
        "model_name": bb.get("model_name"),
        "image_size": prep.get("image_size"),
        "source_recipe": ck.get("source_recipe") if isinstance(ck, dict) else None,
        "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
    }


def _verify_shared(config: dict, skip: bool) -> None:
    if skip:
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "25b_verify_shared_validation.py"),
        "--cv-root", str(config["paths"]["cv"]),
        "--spec", str(config["cv"]["shared_validation_spec"]),
    ]
    print("[VERIFY]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate ViT, ConvNeXt and fixed-weight global ensemble on immutable V5 shared validation"
    )
    ap.add_argument("--config", default="configs/full_cv_pipeline.yaml")
    ap.add_argument("--vit-checkpoint", required=True)
    ap.add_argument("--convnext-checkpoint", required=True)
    ap.add_argument("--convnext-weight", type=float, default=0.45,
                    help="Fixed ConvNeXt score weight. ViT weight = 1-w. Default: 0.45.")
    ap.add_argument("--out", default="artifacts/shared_validation_global_ensemble")
    ap.add_argument("--device", default="0")
    ap.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--image-size", nargs=2, type=int, default=None, metavar=("H", "W"),
                    help="Optional common preprocessing override. Normally omit for v0.9.4 384x576 checkpoints.")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip-split-verification", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    config = _read_yaml(cfg_path)

    vit_ckpt = Path(args.vit_checkpoint).expanduser().resolve()
    cn_ckpt = Path(args.convnext_checkpoint).expanduser().resolve()
    for p in (vit_ckpt, cn_ckpt):
        if not p.is_file():
            raise FileNotFoundError(p)

    wcn = float(args.convnext_weight)
    if not 0.0 <= wcn <= 1.0:
        raise ValueError("--convnext-weight must be in [0,1]")
    wvit = 1.0 - wcn

    _verify_shared(config, args.skip_split_verification)

    spec_path = Path(config["cv"]["shared_validation_spec"])
    if not spec_path.is_absolute():
        spec_path = ROOT / spec_path
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    n_folds = int(spec["official_protocol"]["folds"])
    shared_root = Path(config["paths"]["cv"]) / "shared_validation"

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "evaluation_only": True,
        "training_or_refit_performed": False,
        "benchmark": "vehicle_reid_v5_official shared validation",
        "ensemble": {
            "method": "score_equivalent_weighted_concatenation",
            "vit_weight": wvit,
            "convnext_weight": wcn,
            "formula": "concat(sqrt(w_vit)*L2(z_vit), sqrt(w_convnext)*L2(z_convnext))",
            "weight_selection": "fixed before shared-validation evaluation",
        },
        "vit": _checkpoint_meta(vit_ckpt),
        "convnext": _checkpoint_meta(cn_ckpt),
        "validation_rows": spec["dataset_expectations"]["shared_validation_rows"],
        "validation_vehicle_ids": spec["dataset_expectations"]["shared_validation_vehicle_ids"],
        "validation_spec": str(config["cv"]["shared_validation_spec"]),
    }

    # Warn, but do not silently alter preprocessing.
    vsz = meta["vit"].get("image_size")
    csz = meta["convnext"].get("image_size")
    if args.image_size is None and vsz is not None and csz is not None and list(vsz) != list(csz):
        raise RuntimeError(
            f"Checkpoint preprocessing sizes differ: ViT={vsz}, ConvNeXt={csz}. "
            "Use --image-size H W only if a common size is intentionally desired."
        )

    rows: list[dict] = []
    reports_by_model: dict[str, list[dict]] = {"vit": [], "convnext": [], "ensemble": []}
    resume = not args.no_resume

    for fold in range(n_folds):
        fd = shared_root / f"fold_{fold}"
        qcsv, gcsv, gt = fd / "query.csv", fd / "gallery.csv", fd / "ground_truth.csv"
        for p in (qcsv, gcsv, gt):
            if not p.is_file():
                raise FileNotFoundError(f"Missing shared-validation artifact: {p}")

        fo = out / f"fold_{fold}"
        fo.mkdir(parents=True, exist_ok=True)

        paths = {
            "vit_q": fo / "vit_query.npz",
            "vit_g": fo / "vit_gallery.npz",
            "cn_q": fo / "convnext_query.npz",
            "cn_g": fo / "convnext_gallery.npz",
        }

        jobs = [
            (qcsv, vit_ckpt, paths["vit_q"]),
            (gcsv, vit_ckpt, paths["vit_g"]),
            (qcsv, cn_ckpt, paths["cn_q"]),
            (gcsv, cn_ckpt, paths["cn_g"]),
        ]
        for manifest, checkpoint, cache_path in jobs:
            if resume and cache_path.exists():
                print("[SKIP]", cache_path, flush=True)
                continue
            print("[EXTRACT]", checkpoint.name, manifest, "->", cache_path, flush=True)
            extract_feature_cache(
                manifest, checkpoint, cache_path,
                device=args.device,
                precision=args.precision,
                batch_size=args.batch,
                workers=args.workers,
                image_size=args.image_size,
            )

        vq, vg = load_cache(paths["vit_q"]), load_cache(paths["vit_g"])
        cq, cg = load_cache(paths["cn_q"]), load_cache(paths["cn_g"])

        qids_v, gids_v = _pick_ids(vq, qcsv), _pick_ids(vg, gcsv)
        qids_c, gids_c = _pick_ids(cq, qcsv), _pick_ids(cg, gcsv)
        if qids_v != qids_c:
            raise RuntimeError(f"Fold {fold}: ViT/ConvNeXt query row order mismatch")
        if gids_v != gids_c:
            raise RuntimeError(f"Fold {fold}: ViT/ConvNeXt gallery row order mismatch")

        embeddings = {
            "vit": (
                _l2(vq["z_global"].astype(np.float32)),
                _l2(vg["z_global"].astype(np.float32)),
            ),
            "convnext": (
                _l2(cq["z_global"].astype(np.float32)),
                _l2(cg["z_global"].astype(np.float32)),
            ),
        }
        embeddings["ensemble"] = (
            _ensemble(embeddings["vit"][0], embeddings["convnext"][0], wcn),
            _ensemble(embeddings["vit"][1], embeddings["convnext"][1], wcn),
        )

        # Persist ensemble embeddings for audit/reuse.
        np.savez_compressed(
            fo / "ensemble_embeddings.npz",
            query=embeddings["ensemble"][0].astype(np.float32),
            gallery=embeddings["ensemble"][1].astype(np.float32),
            query_image_id=np.asarray(qids_v, dtype=str),
            gallery_image_id=np.asarray(gids_v, dtype=str),
            vit_weight=np.asarray([wvit], dtype=np.float32),
            convnext_weight=np.asarray([wcn], dtype=np.float32),
        )

        for model_name, (qz, gz) in embeddings.items():
            rep = official_metrics_from_embeddings(
                qz, gz,
                q_ids=qids_v,
                g_ids=gids_v,
                gt_csv=gt,
            )
            reports_by_model[model_name].append(rep)
            row = _metric_row(model_name, fold, rep)
            rows.append(row)
            (fo / f"{model_name}_report.json").write_text(
                json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics_by_fold.csv", index=False)

    metrics = ["mAP@10", "Rank-1", "Rank-5", "mAP_full", "mINP"]
    summary_rows = []
    summary = {**meta, "models": {}}
    for model_name in ("vit", "convnext", "ensemble"):
        md = df[df["model"] == model_name]
        record = {"model": model_name}
        model_summary = {"per_fold": md.to_dict(orient="records")}
        for m in metrics:
            vals = pd.to_numeric(md[m], errors="coerce").dropna().to_numpy(float)
            mean = float(vals.mean()) if len(vals) else float("nan")
            std = float(vals.std(ddof=0)) if len(vals) else float("nan")
            record[m + "_mean"] = mean
            record[m + "_std"] = std
            model_summary[m + "_mean"] = mean
            model_summary[m + "_std"] = std
        summary_rows.append(record)
        summary["models"][model_name] = model_summary

    comp = pd.DataFrame(summary_rows)
    comp.to_csv(out / "comparison.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== GLOBAL BASELINE / ENSEMBLE SHARED VALIDATION ===")
    print(comp.to_string(index=False))
    print(f"\nSaved: {out / 'comparison.csv'}")
    print(f"Saved: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
