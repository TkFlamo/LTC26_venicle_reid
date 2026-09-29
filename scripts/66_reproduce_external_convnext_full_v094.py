#!/usr/bin/env python3
from __future__ import annotations

"""Reproduce the external_convnext_full downstream branch from the developer's raw
ConvNeXt-Small target checkpoint.

This script does NOT claim to recreate the developer's raw checkpoint itself. It
reproduces the complete v0.9.4 downstream stages (parts, part-aware/detail,
reranker, retrieval recipe, refusal calibration, final dev refit and fixed shared
validation) from that checkpoint.
"""

import argparse
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("v094_pipeline", ROOT / "scripts/30_full_cv_pipeline.py")
M = _load("v094_mixed", ROOT / "scripts/51_build_mixed_ensemble_v094.py")

from vehicle_fingerprint.official_validation import generate_official_artifacts, run_official_script


def _rooted(value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def _prepare_pipe(config: Path, data_project: Path, run_root: Path, deploy: Path) -> dict:
    pipe = P.read_yaml(config)
    pipe = copy.deepcopy(pipe)
    pipe["paths"]["raw_csv"] = str(data_project / "raw/train.csv")
    pipe["paths"]["raw_images"] = str(data_project / "raw/images")
    pipe["paths"]["cv"] = str(data_project / "data/processed/hackathon_single_v5shared")
    pipe["paths"]["carparts"] = str(data_project / "data/external/carparts-seg")
    pipe["paths"]["part_bootstrap"] = str(data_project / "data/processed/part_bootstrap")
    pipe["paths"]["runs"] = str(run_root)
    pipe["paths"]["deploy"] = str(deploy)
    return pipe


def _final_shared_validation_external(pipe: dict, args, summary: dict) -> dict:
    out = Path(pipe["paths"]["runs"]) / "final_shared_validation"
    out.mkdir(parents=True, exist_ok=True)
    fit = M.fit_final_external(pipe, args, summary)
    recipe = summary["recipe"]
    refusal = json.loads(Path(summary["refusal"]).read_text(encoding="utf-8"))
    reranker = summary.get("reranker")
    reports = []
    for i, fd in enumerate(P._shared_validation_fold_paths(pipe)):
        fo = out / f"fold_{i}"
        fo.mkdir(parents=True, exist_ok=True)
        q, g = fo / "query.npz", fo / "gallery.npz"
        P.run_cmd(P.extract_cmd(fd / "query.csv", Path(fit["checkpoint"]), q, pipe), log=fo / "features.log", marker=q, resume=args.resume, dry=False, keep_going=False)
        P.run_cmd(P.extract_cmd(fd / "gallery.csv", Path(fit["checkpoint"]), g, pipe), log=fo / "features.log", marker=g, resume=args.resume, dry=False, keep_going=False)
        gt = fd / "ground_truth.csv"
        generate_official_artifacts(q, g, gt, fo, recipe, refusal, reranker_path=reranker, device="cuda", evaluator_path="official/evaluate.py")
        rep = run_official_script(
            gt, fo / "submission.csv", candidates=fo / "candidates.csv", embeddings=fo / "embeddings.npy",
            query_csv=fd / "query.csv", gallery_csv=fd / "gallery.csv", json_out=fo / "official_report.json",
            evaluator_path="official/evaluate.py",
        )
        reports.append(rep)
    df, report = P._aggregate_official_reports(reports)
    df.to_csv(out / "shared_validation_metrics.csv", index=False)
    report.update({
        "benchmark": "vehicle_reid_v5_official shared validation",
        "selected_backbone": summary["backbone"],
        "selected_stage": summary["selected_stage"],
        "selected_initialization": "external_global",
        "external_global_checkpoint": summary.get("global_checkpoint"),
        "selection_leakage_policy": "shared validation reporting-only; all stage/recipe/reranker/refusal selection uses development split",
        "hidden_hackathon_test_used": False,
    })
    (out / "shared_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "development_selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    P.export_deployment(pipe, args, summary, fit, out)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce external_convnext_full downstream training from a raw developer ConvNeXt-Small checkpoint")
    ap.add_argument("--data-project", default=".", help="Project/data root containing raw/ and data/processed")
    ap.add_argument("--convnext-external", required=True, help="Developer raw target-trained ConvNeXt-Small checkpoint (e.g. convnext_best_map.pt)")
    ap.add_argument("--config", default="configs/full_cv_pipeline.yaml")
    ap.add_argument("--run-root", default="runs/reproduce_external_convnext_full_v094")
    ap.add_argument("--deploy", default="deploy/models_v094_external_global")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    a.resume = not a.no_resume
    a.dry = False
    a.keep_going = False

    data_project = Path(a.data_project).expanduser().resolve()
    config = _rooted(a.config)
    run_root = _rooted(a.run_root)
    deploy = _rooted(a.deploy)
    run_root.mkdir(parents=True, exist_ok=True)
    pipe = _prepare_pipe(config, data_project, run_root, deploy)

    # Ensure exact shared-validation split exists and has the expected fingerprint.
    subprocess.run([
        sys.executable, str(ROOT / "scripts/25b_verify_shared_validation.py"),
        "--cv-root", pipe["paths"]["cv"], "--spec", str(_rooted(pipe["cv"]["shared_validation_spec"])),
    ], cwd=ROOT, check=True)

    raw = Path(a.convnext_external).expanduser().resolve()
    if not raw.is_file():
        raise FileNotFoundError(raw)
    prepared = M.prep_external(raw, run_root / "inputs/convnext_small_external_project.pt")
    global_sel = M.gs(prepared, "convnext_small", "external_global", external=raw)

    if a.check_only:
        print(json.dumps({
            "status": "OK",
            "data_project": str(data_project),
            "config": str(config),
            "raw_external_checkpoint": str(raw),
            "prepared_external_checkpoint": str(prepared),
            "cv_root": pipe["paths"]["cv"],
            "carparts": pipe["paths"]["carparts"],
            "deploy": str(deploy),
            "note": "The raw checkpoint can be recreated by scripts/67_train_external_convnext_best_map_v094.py from the vendored developer trainer; this preflight verifies the downstream v0.9.4 stage.",
        }, ensure_ascii=False, indent=2))
        return

    summary = P.advanced_single(pipe, a, global_sel)
    (run_root / "development_winner.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = _final_shared_validation_external(pipe, a, summary)

    # Make provenance explicit in the resulting single deployment.
    dep_meta_path = deploy / "deployment.json"
    dep_meta = json.loads(dep_meta_path.read_text(encoding="utf-8"))
    dep_meta["initialization"] = "external_global"
    dep_meta["source_external_checkpoint"] = str(raw)
    dep_meta["reproducibility"] = {
        "level": "from_data_and_public_pretraining",
        "raw_checkpoint_required_for_this_stage": True,
        "raw_checkpoint_reproducer": "scripts/67_train_external_convnext_best_map_v094.py",
        "vendored_raw_trainer": "third_party/external_convnext_v5_training",
        "downstream_training_reproduced": True,
        "raw_checkpoint_training_code_in_this_project": True,
    }
    dep_meta_path.write_text(json.dumps(dep_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] external_convnext_full deployment: {deploy}")


if __name__ == "__main__":
    main()
