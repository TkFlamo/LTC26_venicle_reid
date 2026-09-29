#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate one or more *already trained* v0.9.4 deployment bundles on the immutable
vehicle_reid_v5 shared validation only.

This script NEVER trains/refits/selects on the shared validation.  It reproduces the
final inference/evaluation path used by scripts/30_full_cv_pipeline.py:

    reid.pt -> feature extraction -> retrieval recipe -> optional pair reranker
            -> refusal -> organizer artifacts -> unchanged official evaluator

Example
-------
python scripts/47_eval_deployments_shared_v5.py \
  --deployment mine=/path/to/models_v094_v5exact_base_veri \
  --deployment other=/path/to/models_v094_external_global \
  --out artifacts/shared_validation_compare \
  --device 0 --precision bf16 --batch 24
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.features import extract_feature_cache
from vehicle_fingerprint.official_validation import generate_official_artifacts, run_official_script


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    if not s:
        raise ValueError("Empty deployment name")
    return s


def parse_deployment_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--deployment must be NAME=/path/to/deployment_dir, got: {raw!r}"
        )
    name, path = raw.split("=", 1)
    return safe_name(name), Path(path).expanduser().resolve()


def resolve_from_metadata(root: Path, metadata: dict, key: str, fallback: str, *, optional: bool = False) -> Path | None:
    value = metadata.get(key)
    if value in (None, "", False):
        if optional:
            p = root / fallback
            return p if p.exists() else None
        value = fallback
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def load_bundle(name: str, root: Path) -> dict:
    if not root.exists():
        raise FileNotFoundError(f"Deployment directory does not exist: {root}")

    meta_path = root / "deployment.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    checkpoint = resolve_from_metadata(root, metadata, "checkpoint", "reid.pt")
    reranker = resolve_from_metadata(root, metadata, "reranker", "reranker.pt", optional=True)
    refusal = resolve_from_metadata(root, metadata, "refusal", "refusal.json")
    recipe = resolve_from_metadata(root, metadata, "retrieval_recipe", "retrieval_recipe.json")

    required = {"checkpoint": checkpoint, "refusal": refusal, "recipe": recipe}
    missing = [f"{k}={v}" for k, v in required.items() if v is None or not Path(v).exists()]
    if missing:
        raise FileNotFoundError(f"Deployment {name!r} is incomplete: " + ", ".join(missing))
    if reranker is not None and not reranker.exists():
        raise FileNotFoundError(f"Deployment {name!r} declares missing reranker: {reranker}")

    return {
        "name": name,
        "root": root,
        "metadata": metadata,
        "checkpoint": Path(checkpoint),
        "reranker": reranker,
        "refusal": Path(refusal),
        "recipe": Path(recipe),
    }


def aggregate_reports(reports: list[dict]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for fold, rep in enumerate(reports):
        r = rep.get("ranking", {})
        fr = rep.get("full_ranking", {})
        c = rep.get("candidates", {})
        rows.append({
            "fold": fold,
            "mAP@10": r.get("mAP@10"),
            "Rank-1": r.get("Rank-1"),
            "Rank-5": r.get("Rank-5"),
            "mAP_full": fr.get("mAP_full"),
            "mINP": fr.get("mINP"),
            "Precision": c.get("Precision"),
            "Recall": c.get("Recall"),
            "F1": c.get("F1"),
            "TNR": c.get("TNR"),
            "PR-AUC": c.get("PR-AUC"),
            "ranking_queries": r.get("n_scored"),
            "openset_queries": r.get("n_openset_excluded"),
        })
    df = pd.DataFrame(rows)
    summary = {"folds": len(rows), "per_fold": rows}
    metrics = ["mAP@10", "Rank-1", "Rank-5", "mAP_full", "mINP", "Precision", "Recall", "F1", "TNR", "PR-AUC"]
    for metric in metrics:
        vals = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy(float)
        summary[metric + "_mean"] = float(vals.mean()) if len(vals) else float("nan")
        summary[metric + "_std"] = float(vals.std()) if len(vals) else float("nan")
    return df, summary


def verify_shared_validation(config: dict, *, skip: bool) -> None:
    if skip:
        return
    cv_root = Path(config["paths"]["cv"])
    spec = Path(config["cv"]["shared_validation_spec"])
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "25b_verify_shared_validation.py"),
        "--cv-root", str(cv_root),
        "--spec", str(spec),
    ]
    print("[VERIFY]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def evaluate_bundle(bundle: dict, *, config: dict, out_root: Path, device: str, precision: str,
                    batch: int, workers: int, resume: bool) -> dict:
    name = bundle["name"]
    dst = out_root / name
    dst.mkdir(parents=True, exist_ok=True)

    recipe = json.loads(bundle["recipe"].read_text(encoding="utf-8"))
    refusal = json.loads(bundle["refusal"].read_text(encoding="utf-8"))
    reranker = bundle["reranker"]

    spec_path = Path(config["cv"]["shared_validation_spec"])
    if not spec_path.is_absolute():
        spec_path = ROOT / spec_path
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    n_folds = int(spec["official_protocol"]["folds"])
    shared_root = Path(config["paths"]["cv"]) / "shared_validation"

    reports: list[dict] = []
    for fold in range(n_folds):
        fd = shared_root / f"fold_{fold}"
        for required in (fd / "query.csv", fd / "gallery.csv", fd / "ground_truth.csv"):
            if not required.exists():
                raise FileNotFoundError(f"Missing shared-validation artifact: {required}")

        fo = dst / f"fold_{fold}"
        fo.mkdir(parents=True, exist_ok=True)
        q_cache = fo / "query.npz"
        g_cache = fo / "gallery.npz"

        print(f"\n[{name}] fold {fold}/{n_folds - 1}: feature extraction", flush=True)
        if not (resume and q_cache.exists()):
            extract_feature_cache(
                fd / "query.csv", bundle["checkpoint"], q_cache,
                device=device, precision=precision, batch_size=batch,
                workers=workers,
            )
        else:
            print(f"[SKIP] {q_cache}", flush=True)

        if not (resume and g_cache.exists()):
            extract_feature_cache(
                fd / "gallery.csv", bundle["checkpoint"], g_cache,
                device=device, precision=precision, batch_size=batch,
                workers=workers,
            )
        else:
            print(f"[SKIP] {g_cache}", flush=True)

        official_report = fo / "official_report.json"
        if resume and official_report.exists():
            rep = json.loads(official_report.read_text(encoding="utf-8"))
            print(f"[SKIP] {official_report}", flush=True)
        else:
            gt = fd / "ground_truth.csv"
            generate_official_artifacts(
                q_cache, g_cache, gt, fo, recipe, refusal,
                reranker_path=reranker,
                device="cuda" if str(device) not in {"cpu", "-1"} else "cpu",
                evaluator_path="official/evaluate.py",
            )
            rep = run_official_script(
                gt,
                fo / "submission.csv",
                candidates=fo / "candidates.csv",
                embeddings=fo / "embeddings.npy",
                query_csv=fd / "query.csv",
                gallery_csv=fd / "gallery.csv",
                json_out=official_report,
                evaluator_path="official/evaluate.py",
            )
        reports.append(rep)

    metrics_df, report = aggregate_reports(reports)
    metrics_df.to_csv(dst / "shared_validation_metrics.csv", index=False)
    report.update({
        "benchmark": "vehicle_reid_v5_official shared validation",
        "evaluation_only": True,
        "training_or_refit_performed": False,
        "deployment_name": name,
        "deployment_root": str(bundle["root"]),
        "checkpoint": str(bundle["checkpoint"]),
        "reranker": str(reranker) if reranker else None,
        "retrieval_recipe": str(bundle["recipe"]),
        "refusal": str(bundle["refusal"]),
        "selected_backbone": bundle["metadata"].get("backbone"),
        "selected_stage": bundle["metadata"].get("selected_stage"),
        "initialization": bundle["metadata"].get("initialization"),
        "validation_rows": spec["dataset_expectations"]["shared_validation_rows"],
        "validation_vehicle_ids": spec["dataset_expectations"]["shared_validation_vehicle_ids"],
        "development_rows": spec["dataset_expectations"]["development_rows"],
        "development_vehicle_ids": spec["dataset_expectations"]["development_vehicle_ids"],
        "selection_leakage_policy": "reporting-only shared validation; no fitting or tuning performed by this script",
        "validation_spec": str(config["cv"]["shared_validation_spec"]),
    })
    (dst / "shared_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[{name}] FIXED SHARED VALIDATION")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate trained v0.9.4 deployments on the immutable shared V5 validation")
    ap.add_argument("--config", default="configs/full_cv_pipeline.yaml")
    ap.add_argument(
        "--deployment", action="append", required=True, metavar="NAME=DIR",
        help="Repeatable. Deployment dir must contain reid.pt, retrieval_recipe.json, refusal.json and optional reranker.pt/deployment.json",
    )
    ap.add_argument("--out", default="artifacts/shared_validation_compare")
    ap.add_argument("--device", default="0")
    ap.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip-split-verification", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = read_yaml(config_path)

    verify_shared_validation(config, skip=args.skip_split_verification)

    deployments = [load_bundle(*parse_deployment_arg(x)) for x in args.deployment]
    names = [x["name"] for x in deployments]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate deployment names: {names}")

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    reports = []
    for bundle in deployments:
        rep = evaluate_bundle(
            bundle,
            config=config,
            out_root=out_root,
            device=args.device,
            precision=args.precision,
            batch=args.batch,
            workers=args.workers,
            resume=not args.no_resume,
        )
        reports.append(rep)

    fields = [
        "deployment_name", "selected_backbone", "selected_stage", "initialization",
        "mAP@10_mean", "mAP@10_std", "Rank-1_mean", "Rank-1_std",
        "Rank-5_mean", "Rank-5_std", "mAP_full_mean", "mAP_full_std",
        "mINP_mean", "mINP_std", "Precision_mean", "Recall_mean", "F1_mean", "TNR_mean", "PR-AUC_mean",
    ]
    comparison = pd.DataFrame([{k: r.get(k) for k in fields} for r in reports])
    comparison.to_csv(out_root / "comparison.csv", index=False)
    (out_root / "comparison.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== SHARED VALIDATION COMPARISON ===")
    print(comparison.to_string(index=False))
    print(f"\nSaved: {out_root / 'comparison.csv'}")


if __name__ == "__main__":
    main()
