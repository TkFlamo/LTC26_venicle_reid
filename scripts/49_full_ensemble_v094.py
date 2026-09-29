#!/usr/bin/env python3
from __future__ import annotations

"""
Build and evaluate a FULL v0.9.4 two-backbone ensemble without using shared
validation for any training, recipe selection, reranker fitting, refusal
calibration, epoch selection or ensemble-weight tuning.

Flow
----
fixed project global checkpoints (ConvNeXt + ViT)
  -> full branch A: parts -> part-aware -> detail (development only)
  -> full branch B: parts -> part-aware -> detail (development only)
  -> fixed score-equivalent ensemble (default ConvNeXt=0.45, ViT=0.55)
  -> ensemble-specific reranker on inner/reranker_fit
  -> recipe + refusal on inner/selection_official
  -> refit BOTH selected branches on all development identities
  -> evaluate immutable 4-fold shared validation (reporting only)
  -> export deploy/models_v094_full_ensemble/

The hidden hackathon test is never read by this script.
"""

import argparse
import copy
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.cv_experiments import (
    checkpoint_official_metrics,
    fit_pooled_refusal,
    select_recipe_across_folds,
)
from vehicle_fingerprint.features import ensemble_feature_caches
from vehicle_fingerprint.official_validation import (
    evaluate_recipe,
    generate_official_artifacts,
    run_official_script,
)


def _load_pipeline_module():
    p = ROOT / "scripts" / "30_full_cv_pipeline.py"
    spec = importlib.util.spec_from_file_location("v094_pipeline", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_pipeline_module()


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rooted(s: str | Path) -> Path:
    p = Path(s).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def _selected_representation(summary: dict) -> dict:
    stage_name = str(summary["selected_stage"])
    candidates = [x for x in summary["stages"] if str(x.get("stage")) == stage_name]
    if candidates:
        return candidates[0]
    # listwise stage stores the underlying detail checkpoint/cache in the stage row.
    if stage_name == "listwise_reranker_kreciprocal":
        candidates = [x for x in summary["stages"] if str(x.get("stage")) == "detail_tuning"]
        if candidates:
            return candidates[0]
    raise RuntimeError(f"Cannot resolve selected representation for stage={stage_name}")


def _global_sel(checkpoint: Path, backbone: str) -> dict:
    m = checkpoint_official_metrics(checkpoint)
    return {
        "backbone": backbone,
        "family": "convnext" if backbone.startswith("convnext") else "vit",
        "resolution": "384x576",
        "initialization": "dinov3_direct",
        "init_checkpoint": None,
        "veri_pretrain_checkpoint": None,
        **m,
        "checkpoint": str(checkpoint),
        "native_checkpoint": None,
    }


def _extract(pipe: dict, args, manifest: Path, checkpoint: Path, out: Path, log: Path):
    P.run_cmd(
        P.extract_cmd(manifest, checkpoint, out, pipe),
        log=log, marker=out, resume=args.resume, dry=False, keep_going=False,
    )
    return out


def _run(cmd: list[str | Path], *, log: Path, marker: Path | None, args):
    return P.run_cmd(
        cmd, log=log, marker=marker, resume=args.resume, dry=False, keep_going=False
    )


def _make_branch_pipe(pipe: dict, run_root: Path) -> dict:
    q = copy.deepcopy(pipe)
    q["paths"]["runs"] = str(run_root)
    # Deployment is handled only once, after ensemble shared-validation.
    q["paths"]["deploy"] = str(run_root / "_unused_deploy")
    return q


def _fit_ensemble_dev(
    pipe: dict,
    args,
    *,
    cn_summary: dict,
    vit_summary: dict,
    root: Path,
    convnext_weight: float,
) -> dict:
    inner = Path(pipe["paths"]["cv"]) / "inner"
    rr = pipe["reranker"]
    root.mkdir(parents=True, exist_ok=True)

    cn_rep = _selected_representation(cn_summary)
    vit_rep = _selected_representation(vit_summary)
    cn_ckpt = Path(cn_rep["checkpoint"])
    vit_ckpt = Path(vit_rep["checkpoint"])

    # 1) Ensemble-specific reranker-fit cache.
    rrfit = root / "reranker_fit"
    cn_rr = rrfit / "convnext.npz"
    vi_rr = rrfit / "vit.npz"
    ens_rr = rrfit / "ensemble.npz"
    _extract(pipe, args, inner / "reranker_fit.csv", cn_ckpt, cn_rr, rrfit / "features_convnext.log")
    _extract(pipe, args, inner / "reranker_fit.csv", vit_ckpt, vi_rr, rrfit / "features_vit.log")
    if not (args.resume and ens_rr.exists()):
        ensemble_feature_caches(cn_rr, vi_rr, ens_rr, weight_a=convnext_weight)

    hard = rrfit / "hard.json"
    _run([
        sys.executable, "scripts/12_mine_hard_negatives.py",
        "--cache", ens_rr,
        "--out", hard,
        "--topk", "30",
        "--representation", "global",
        "--refine-factor", "3",
        "--top-pair-mean", "2",
    ], log=rrfit / "mine.log", marker=hard, args=args)

    pairs = rrfit / "pairs.npz"
    _run([
        sys.executable, "scripts/28_build_reranker_pairs.py",
        "--cache", ens_rr,
        "--hard-map", hard,
        "--out", pairs,
        "--mode", str(rr["mode"]),
        "--candidates-per-query", str(rr["candidates_per_query"]),
        "--groups-per-id", str(rr["groups_per_id"]),
        "--knn-k", str(rr["knn_k"]),
    ], log=rrfit / "pairs.log", marker=pairs, args=args)

    reranker = root / "reranker" / "best.pt"
    _run([
        sys.executable, "scripts/29_train_reranker_from_pairs.py",
        "--pairs", pairs,
        "--out", reranker,
        "--epochs", str(rr["epochs"]),
        "--device", "cuda",
        "--mode", str(rr["mode"]),
    ], log=root / "reranker" / "train.log", marker=reranker, args=args)

    # 2) Recipe selection on the dedicated development selection protocol.
    selection = inner / "selection_official"
    sq_cn = root / "selection" / "convnext_query.npz"
    sg_cn = root / "selection" / "convnext_gallery.npz"
    sq_vi = root / "selection" / "vit_query.npz"
    sg_vi = root / "selection" / "vit_gallery.npz"
    sq = root / "selection" / "ensemble_query.npz"
    sg = root / "selection" / "ensemble_gallery.npz"

    _extract(pipe, args, selection / "query.csv", cn_ckpt, sq_cn, root / "selection" / "features_cn.log")
    _extract(pipe, args, selection / "gallery.csv", cn_ckpt, sg_cn, root / "selection" / "features_cn.log")
    _extract(pipe, args, selection / "query.csv", vit_ckpt, sq_vi, root / "selection" / "features_vit.log")
    _extract(pipe, args, selection / "gallery.csv", vit_ckpt, sg_vi, root / "selection" / "features_vit.log")
    if not (args.resume and sq.exists()):
        ensemble_feature_caches(sq_cn, sq_vi, sq, weight_a=convnext_weight)
    if not (args.resume and sg.exists()):
        ensemble_feature_caches(sg_cn, sg_vi, sg, weight_a=convnext_weight)

    desc = [{
        "query_cache": str(sq),
        "gallery_cache": str(sg),
        "gt": str(selection / "ground_truth.csv"),
        "reranker": str(reranker),
    }]

    # Preserve the old canonical ensemble contract: retrieval itself is global
    # ensemble (alpha=0); local/part evidence is available to the reranker.
    recipe, search = select_recipe_across_folds(
        desc,
        alpha_grid=[0.0],
        beta_grid=rr["beta_grid"],
        kreciprocal_grid=rr["kreciprocal_lambda_grid"],
        same_camera_grid=rr["same_camera_filter_grid"],
        rerank_topk=rr["rerank_topk"],
        kreciprocal_k=rr["kreciprocal_k"],
        device="cuda",
    )
    search.to_csv(root / "retrieval_recipe_search.csv", index=False)
    (root / "retrieval_recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    refusal = root / "refusal_selection.json"
    fit_pooled_refusal(desc, recipe, refusal, device="cuda")

    rep, *_ = evaluate_recipe(
        sq, sg, selection / "ground_truth.csv",
        base_alpha=recipe["base_alpha"],
        reranker_path=reranker,
        reranker_beta=recipe["reranker_beta"],
        rerank_topk=recipe["rerank_topk"],
        kreciprocal_lambda=recipe["kreciprocal_lambda"],
        kreciprocal_k=recipe["kreciprocal_k"],
        same_camera_filter=recipe["same_camera_filter"],
        device="cuda",
        evaluator_path="official/evaluate.py",
    )

    r, fr = rep["ranking"], rep["full_ranking"]
    summary = {
        "family": "ensemble",
        "selected_stage": "full_ensemble_reranker_kreciprocal",
        "convnext_weight": float(convnext_weight),
        "vit_weight": float(1.0 - convnext_weight),
        "recipe": recipe,
        "reranker": str(reranker),
        "refusal": str(refusal),
        "development_selection_metrics": {
            "mAP@10": float(r["mAP@10"]),
            "Rank-1": float(r["Rank-1"]),
            "Rank-5": float(r["Rank-5"]),
            "mAP_full": float(fr["mAP_full"]),
            "mINP": float(fr["mINP"]),
        },
        "convnext": cn_summary,
        "vit": vit_summary,
        "selection_leakage_policy": (
            "ensemble weight fixed before shared validation; branch/stage epochs, "
            "reranker, retrieval recipe and refusal fitted only on development"
        ),
    }
    (root / "ensemble_development_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _aggregate(reports: list[dict]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for fold, rep in enumerate(reports):
        r, fr, c = rep.get("ranking", {}), rep.get("full_ranking", {}), rep.get("candidates", {})
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
    out = {"folds": len(rows), "per_fold": rows}
    for m in ["mAP@10","Rank-1","Rank-5","mAP_full","mINP","Precision","Recall","F1","TNR","PR-AUC"]:
        v = pd.to_numeric(df[m], errors="coerce").dropna().to_numpy(float)
        out[m + "_mean"] = float(v.mean()) if len(v) else float("nan")
        out[m + "_std"] = float(v.std()) if len(v) else float("nan")
    return df, out


def _export_deployment(
    pipe: dict,
    args,
    *,
    cn_fit: dict,
    vit_fit: dict,
    ensemble_summary: dict,
    shared_dir: Path,
    deploy: Path,
):
    deploy.mkdir(parents=True, exist_ok=True)
    cn_dst = deploy / "reid_convnext.pt"
    vi_dst = deploy / "reid_vit.pt"
    _run([
        sys.executable, "scripts/16_export_inference_checkpoint.py",
        "--src", cn_fit["checkpoint"], "--out", cn_dst,
    ], log=deploy / "export.log", marker=cn_dst, args=args)
    _run([
        sys.executable, "scripts/16_export_inference_checkpoint.py",
        "--src", vit_fit["checkpoint"], "--out", vi_dst,
    ], log=deploy / "export.log", marker=vi_dst, args=args)

    rr_dst = deploy / "reranker.pt"
    shutil.copy2(ensemble_summary["reranker"], rr_dst)
    shutil.copy2(ensemble_summary["refusal"], deploy / "refusal.json")
    (deploy / "retrieval_recipe.json").write_text(
        json.dumps(ensemble_summary["recipe"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    info = {
        "mode": "ensemble",
        "checkpoint_a": "reid_convnext.pt",
        "checkpoint_b": "reid_vit.pt",
        "checkpoint_a_family": "convnext",
        "checkpoint_b_family": "vit",
        "weight_a": float(ensemble_summary["convnext_weight"]),
        "convnext_weight": float(ensemble_summary["convnext_weight"]),
        "vit_weight": float(ensemble_summary["vit_weight"]),
        "selected_stage": ensemble_summary["selected_stage"],
        "reranker": "reranker.pt",
        "refusal": "refusal.json",
        "retrieval_recipe": "retrieval_recipe.json",
        "shared_validation_report": str(shared_dir / "shared_validation_report.json"),
        "inference_script": "scripts/22_infer_ensemble_from_csv.py",
        "hidden_test_leakage": False,
        "training_data_policy": "all fitting/calibration uses development only",
    }
    (deploy / "deployment.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DEPLOY] {deploy}")


def main():
    ap = argparse.ArgumentParser(
        description="Full v0.9.4 ConvNeXt+ViT ensemble: development fit -> fixed shared validation -> deploy"
    )
    ap.add_argument("--config", default="configs/full_cv_pipeline.yaml")
    ap.add_argument("--convnext-global", required=True,
                    help="v0.9.4 project checkpoint for your ConvNeXt-Base global model")
    ap.add_argument("--vit-global", required=True,
                    help="v0.9.4 project checkpoint for your ViT-Base global model")
    ap.add_argument("--convnext-weight", type=float, default=0.45,
                    help="FIXED before shared validation. Default 0.45; ViT gets 1-w.")
    ap.add_argument("--run-root", default="runs/full_ensemble_v094")
    ap.add_argument("--deploy", default="deploy/models_v094_full_ensemble")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip-data-prepare", action="store_true",
                    help="Use when the immutable dev/shared split and CarParts dataset already exist.")
    a = ap.parse_args()
    a.resume = not a.no_resume
    a.dry = False
    a.keep_going = False

    if not 0.0 <= float(a.convnext_weight) <= 1.0:
        raise SystemExit("--convnext-weight must be in [0,1]")

    cfg = _rooted(a.config)
    pipe = _read_yaml(cfg)
    run_root = _rooted(a.run_root)
    deploy = _rooted(a.deploy)
    run_root.mkdir(parents=True, exist_ok=True)

    cn_global = _rooted(a.convnext_global)
    vit_global = _rooted(a.vit_global)
    for p in (cn_global, vit_global):
        if not p.is_file():
            raise FileNotFoundError(p)

    # Verify the immutable split before doing any work. This does not score it.
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "25b_verify_shared_validation.py"),
        "--cv-root", str(pipe["paths"]["cv"]),
        "--spec", str(pipe["cv"]["shared_validation_spec"]),
    ], cwd=ROOT, check=True)

    if not a.skip_data_prepare:
        prep_pipe = copy.deepcopy(pipe)
        prep_pipe["paths"]["runs"] = str(run_root / "_prepare")
        P.ensure_data(prep_pipe, a)

    # Development-only branch training/selection.
    cn_pipe = _make_branch_pipe(pipe, run_root / "branches" / "convnext")
    vi_pipe = _make_branch_pipe(pipe, run_root / "branches" / "vit")
    cn_summary = P.advanced_single(cn_pipe, a, _global_sel(cn_global, "convnext_base"))
    vit_summary = P.advanced_single(vi_pipe, a, _global_sel(vit_global, "vit_base"))

    ensemble = _fit_ensemble_dev(
        pipe, a,
        cn_summary=cn_summary,
        vit_summary=vit_summary,
        root=run_root / "ensemble_development",
        convnext_weight=float(a.convnext_weight),
    )

    # Final fixed-epoch refit of BOTH branches on all development identities.
    cn_refit_pipe = _make_branch_pipe(pipe, run_root / "final_refit" / "convnext")
    vi_refit_pipe = _make_branch_pipe(pipe, run_root / "final_refit" / "vit")
    cn_fit = P.fit_final(cn_refit_pipe, a, cn_summary)
    vit_fit = P.fit_final(vi_refit_pipe, a, vit_summary)

    # Reporting-only shared validation. Nothing below changes model/recipe/calibration.
    shared = run_root / "final_shared_validation"
    shared.mkdir(parents=True, exist_ok=True)
    refusal = json.loads(Path(ensemble["refusal"]).read_text(encoding="utf-8"))
    recipe = ensemble["recipe"]
    reranker = ensemble["reranker"]
    reports = []

    for i, fd in enumerate(P._shared_validation_fold_paths(pipe)):
        fo = shared / f"fold_{i}"
        fo.mkdir(parents=True, exist_ok=True)

        cnq, cng = fo / "convnext_query.npz", fo / "convnext_gallery.npz"
        viq, vig = fo / "vit_query.npz", fo / "vit_gallery.npz"
        q, g = fo / "ensemble_query.npz", fo / "ensemble_gallery.npz"

        _extract(pipe, a, fd / "query.csv", Path(cn_fit["checkpoint"]), cnq, fo / "features_cn.log")
        _extract(pipe, a, fd / "gallery.csv", Path(cn_fit["checkpoint"]), cng, fo / "features_cn.log")
        _extract(pipe, a, fd / "query.csv", Path(vit_fit["checkpoint"]), viq, fo / "features_vit.log")
        _extract(pipe, a, fd / "gallery.csv", Path(vit_fit["checkpoint"]), vig, fo / "features_vit.log")

        if not (a.resume and q.exists()):
            ensemble_feature_caches(cnq, viq, q, weight_a=float(a.convnext_weight))
        if not (a.resume and g.exists()):
            ensemble_feature_caches(cng, vig, g, weight_a=float(a.convnext_weight))

        gt = fd / "ground_truth.csv"
        generate_official_artifacts(
            q, g, gt, fo, recipe, refusal,
            reranker_path=reranker, device="cuda", evaluator_path="official/evaluate.py",
        )
        rep = run_official_script(
            gt, fo / "submission.csv",
            candidates=fo / "candidates.csv",
            embeddings=fo / "embeddings.npy",
            query_csv=fd / "query.csv",
            gallery_csv=fd / "gallery.csv",
            json_out=fo / "official_report.json",
            evaluator_path="official/evaluate.py",
        )
        reports.append(rep)

    df, report = _aggregate(reports)
    df.to_csv(shared / "shared_validation_metrics.csv", index=False)

    spec_path = Path(pipe["cv"]["shared_validation_spec"])
    if not spec_path.is_absolute():
        spec_path = ROOT / spec_path
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    report.update({
        "benchmark": "vehicle_reid_v5_official shared validation",
        "validation_rows": spec["dataset_expectations"]["shared_validation_rows"],
        "validation_vehicle_ids": spec["dataset_expectations"]["shared_validation_vehicle_ids"],
        "development_rows": spec["dataset_expectations"]["development_rows"],
        "development_vehicle_ids": spec["dataset_expectations"]["development_vehicle_ids"],
        "selected_backbone": "convnext_base+vit_base",
        "selected_stage": ensemble["selected_stage"],
        "selected_initialization": "dinov3_direct_two_branch",
        "convnext_weight": float(a.convnext_weight),
        "vit_weight": float(1.0 - a.convnext_weight),
        "selection_leakage_policy": (
            "shared validation is reporting-only inside this run; ensemble weight fixed in advance; "
            "all epochs/reranker/recipe/refusal fitted on development only"
        ),
        "hidden_hackathon_test_used": False,
        "validation_spec": pipe["cv"]["shared_validation_spec"],
    })
    (shared / "shared_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (shared / "ensemble_development_summary.json").write_text(
        json.dumps(ensemble, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _export_deployment(
        pipe, a, cn_fit=cn_fit, vit_fit=vit_fit,
        ensemble_summary=ensemble, shared_dir=shared, deploy=deploy,
    )

    print("\nFULL ENSEMBLE FIXED SHARED VALIDATION\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
