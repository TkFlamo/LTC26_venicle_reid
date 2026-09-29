#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"base_full": 0.65, "external_convnext_full": 0.35}
WEIGHT_EXTS = {".pt", ".pth", ".bin", ".onnx", ".engine", ".plan", ".safetensors", ".ckpt", ".trt", ".pb", ".tflite", ".npz"}
FORBIDDEN_FINAL_EXTS = {".onnx", ".engine", ".plan", ".trt", ".pb", ".tflite"}
MAX_WEIGHT_BYTES = 2 * 1024**3
BUILTIN_EXTERNAL_TRAINER = ROOT / "third_party" / "external_convnext_v5_training"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_production(meta: dict) -> None:
    if str(meta.get("mode")) != "score_ensemble":
        raise RuntimeError("Expected mode=score_ensemble")
    got = {str(x["name"]): float(x["weight"]) for x in meta.get("members", [])}
    if set(got) != set(EXPECTED) or any(abs(got[k] - v) > 1e-9 for k, v in EXPECTED.items()):
        raise RuntimeError(f"Expected production ensemble {EXPECTED}; got {got}")
    if str(meta.get("fusion")) != "per_query_zscore":
        raise RuntimeError("Expected per_query_zscore fusion")


def _resolve(root: Path, value, fallback=None, optional=False) -> Path | None:
    value = value or fallback
    if not value:
        return None
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if optional and not p.exists():
        return None
    return p


def _copy_member_runtime(src_member: Path, dst_member: Path) -> None:
    meta_path = src_member / "deployment.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = _read_json(meta_path)
    if str(meta.get("mode", "single")).lower() != "single":
        raise RuntimeError(f"Expected single member deployment: {src_member}")

    dst_member.mkdir(parents=True, exist_ok=True)
    runtime_meta = dict(meta)
    refs = (
        ("checkpoint", "reid.pt", False),
        ("retrieval_recipe", "retrieval_recipe.json", False),
        ("refusal", "refusal.json", True),
        ("reranker", "reranker.pt", True),
    )
    for key, fallback, optional in refs:
        src = _resolve(src_member, meta.get(key), fallback, optional=optional)
        if src is None:
            runtime_meta.pop(key, None)
            continue
        if not src.is_file():
            raise FileNotFoundError(src)
        dst_name = src.name
        shutil.copy2(src, dst_member / dst_name)
        runtime_meta[key] = dst_name

    # Strip machine-local training/source paths if present.
    for key in list(runtime_meta):
        if key.startswith("source_") and key not in {"source_mode"}:
            runtime_meta.pop(key, None)
    (dst_member / "deployment.json").write_text(json.dumps(runtime_meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _sanitize_top_deployment(meta: dict) -> dict:
    out = dict(meta)
    out["members"] = [
        {
            "name": str(item["name"]),
            "path": str(Path("members") / str(item["name"])),
            "weight": float(item["weight"]),
            "source_mode": "single",
        }
        for item in meta["members"]
    ]
    out["runtime"] = {
        **(out.get("runtime", {}) or {}),
        "feature_runtime": "pytorch_cuda",
        "weight_format": ".pt",
        "recommended_precision": "fp16",
        "shared_preprocess": True,
    }
    out["final_runtime"] = {
        "feature_runtime": "pytorch_cuda",
        "pytorch_checkpoints_in_final_package": True,
        "onnx_artifacts_in_final_package": False,
        "tensorrt_artifacts_in_final_package": False,
    }
    return out


def _exact_requirement_lines(path: Path) -> list[str]:
    bad = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            bad.append(line)
    return bad


_EXTERNAL_SOURCE_SKIP_DIRS = {
    ".git", ".idea", ".vscode", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".venv", "venv", "env", "runs", "outputs", "artifacts",
    "data", "raw", "deploy", "dist", "wandb", "checkpoints", "weights",
}
_EXTERNAL_SOURCE_SKIP_EXTS = WEIGHT_EXTS | {".npy", ".npz", ".pkl", ".pickle", ".parquet"}


def _copy_training_source_code(src: Path, dst: Path) -> dict:
    """Copy source/config files for the external trainer without datasets or weight blobs."""
    src = src.resolve()
    if not src.is_dir():
        raise NotADirectoryError(src)
    files = []
    total = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        if any(part in _EXTERNAL_SOURCE_SKIP_DIRS for part in rel.parts):
            continue
        if p.suffix.lower() in _EXTERNAL_SOURCE_SKIP_EXTS:
            continue
        # Avoid accidentally copying very large opaque files into the reproducibility source bundle.
        size = p.stat().st_size
        if size > 20 * 1024**2:
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        total += size
        files.append({"path": str(rel), "size_bytes": int(size), "sha256": sha256(p)})
    if not files:
        raise RuntimeError(f"No source/config files copied from external training source: {src}")
    return {"source": str(src), "files": files, "file_count": len(files), "total_bytes": int(total)}


def _audit_builtin_external_trainer() -> dict:
    required = [
        "vehicle_reid_v5.py",
        "train_best_v5.py",
        "run_next_experiments_v7.py",
        "requirements_v5.txt",
        "README_INTEGRATION.md",
        "SOURCE_SHA256.txt",
        "reference/E3e_camera_strongerase_history.csv",
        "reference/E3e_camera_strongerase_split.json",
    ]
    missing = [x for x in required if not (BUILTIN_EXTERNAL_TRAINER / x).is_file()]
    if missing:
        raise RuntimeError(f"Built-in external ConvNeXt training source is incomplete: {missing}")
    files=[]
    total=0
    for rel in required:
        q=BUILTIN_EXTERNAL_TRAINER / rel
        size=q.stat().st_size
        total += size
        files.append({"path": str(Path("third_party/external_convnext_v5_training") / rel), "size_bytes": int(size), "sha256": sha256(q)})
    return {
        "source": "vendored_team_developer_training_source",
        "path": "third_party/external_convnext_v5_training",
        "files": files,
        "file_count": len(files),
        "total_bytes": int(total),
        "raw_checkpoint_entrypoint": "scripts/67_train_external_convnext_best_map_v094.py",
        "winning_profile": "small_camera_sampler_strongerase",
        "reference_run": "E3e_camera_strongerase",
        "reference_best_epoch": 25,
        "reference_best_official_mAP@10": 0.8214526158811873,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Create final PT-only hackathon package with runtime + fully vendored training reproducibility (<2GB inference weights)")
    ap.add_argument("--deployment-dir", default="deploy/models_current")
    ap.add_argument("--out", default="dist/falcon_reid_v094_pt")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-unresolved-provenance", action="store_true")
    ap.add_argument(
        "--external-convnext-seed",
        default=None,
        help=(
            "Optional existing convnext_best_map.pt to package for reproducibility. "
            "If omitted, weights/external/convnext_best_map.pt is auto-detected when present."
        ),
    )
    ap.add_argument(
        "--require-external-convnext-seed",
        action="store_true",
        help="Fail packaging if convnext_best_map.pt cannot be found/provided.",
    )
    a = ap.parse_args()

    dep = Path(a.deployment_dir).expanduser()
    if not dep.is_absolute(): dep = (ROOT / dep).resolve()
    out = Path(a.out).expanduser()
    if not out.is_absolute(): out = (ROOT / out).resolve()

    meta_path = dep / "deployment.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = _read_json(meta_path)
    _validate_production(meta)

    prov = meta.get("provenance", {}) or {}
    ext = prov.get("external_convnext_full", {}) or {}
    origin = str(ext.get("origin", "unresolved"))
    if origin == "public_external":
        missing = [k for k in ("url", "version", "sha256") if not ext.get(k)]
        if missing:
            raise RuntimeError(f"public_external provenance is incomplete; missing {missing}")
    elif origin not in {"team_trained", "public_external"} and not a.allow_unresolved_provenance:
        raise RuntimeError(
            "external_convnext_full provenance is unresolved. Rebuild deployment with "
            "--external-origin team_trained OR public_external (+ URL/version/sha256), "
            "or use --allow-unresolved-provenance only for a non-final test package."
        )

    if out.exists():
        if not a.force: raise SystemExit(f"Output exists: {out}; use --force")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Runtime + full training/reproducibility stack.  Training data and generated runs are NOT copied.
    for d in ("src", "scripts", "official", "configs", "third_party", "tests"):
        if (ROOT / d).is_dir():
            shutil.copytree(ROOT / d, out / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    for name in (
        "pyproject.toml", "requirements-runtime-a5000.txt", "requirements-training.txt", "requirements-training-external-convnext.txt", "requirements.txt", "run_hackathon.py",
        "HACKATHON_COMPLIANCE.md", "PT_RUNTIME_V094.md", "PRODUCTION_MODEL_V094.md",
        "TRAINING_REPRODUCIBILITY.md", "V5_EXACT_BASELINE.md", "EXTERNAL_DATASETS.md",
        "SHARED_VALIDATION_V5.md", "Dockerfile.training", "Dockerfile.external-training", "LICENSE",
    ):
        p = ROOT / name
        if p.is_file(): shutil.copy2(p, out / name)
    shutil.copy2(ROOT / "Dockerfile.runtime", out / "Dockerfile")

    # The raw external ConvNeXt trainer is now vendored directly in this project.
    # No extra --external-training-source argument is needed for a final package.
    training_source_audit = _audit_builtin_external_trainer()

    # Optionally package an already-trained convnext_best_map.pt so the expensive raw
    # ConvNeXt training can be skipped while preserving exact downstream reproducibility.
    seed_src = None
    if a.external_convnext_seed:
        seed_src = Path(a.external_convnext_seed).expanduser()
        if not seed_src.is_absolute():
            seed_src = (ROOT / seed_src).resolve()
    else:
        auto_seed = (ROOT / "weights" / "external" / "convnext_best_map.pt").resolve()
        if auto_seed.is_file():
            seed_src = auto_seed

    packaged_seed = None
    if seed_src is not None:
        if not seed_src.is_file():
            raise FileNotFoundError(seed_src)
        seed_dst = out / "training" / "checkpoints" / "external_convnext" / "convnext_best_map.pt"
        seed_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed_src, seed_dst)
        packaged_seed = {
            "source": str(seed_src),
            "path": str(seed_dst.relative_to(out)),
            "size_bytes": int(seed_dst.stat().st_size),
            "sha256": sha256(seed_dst),
        }
    elif a.require_external_convnext_seed:
        raise RuntimeError(
            "convnext_best_map.pt was required but not found. Pass --external-convnext-seed /path/to/convnext_best_map.pt "
            "or place it at weights/external/convnext_best_map.pt."
        )

    training_repro = {
        "schema": "vehicle-reid-v094-training-reproducibility-v1",
        "production_ensemble": EXPECTED,
        "base_full": {
            "level": "from_data_and_public_pretraining",
            "training_code_included": True,
            "entrypoint": "scripts/30_full_cv_pipeline.py",
            "config": "configs/full_cv_pipeline.yaml",
            "vendored_baseline": "third_party/vehicle_reid_v5_official",
        },
        "external_convnext_full": {
            "level": "from_data_and_public_pretraining",
            "raw_checkpoint_training_source_included": True,
            "raw_checkpoint_training_entrypoint": "scripts/67_train_external_convnext_best_map_v094.py",
            "raw_checkpoint_training_command": "GPU=0 python scripts/67_train_external_convnext_best_map_v094.py --data /path/to/hackathon_dataset",
            "raw_checkpoint_output": "weights/external/convnext_best_map.pt",
            "pretrained_seed_packaged": packaged_seed is not None,
            "packaged_seed": packaged_seed,
            "downstream_training_code_included": True,
            "downstream_entrypoint": "scripts/66_reproduce_external_convnext_full_v094.py",
            "downstream_from_packaged_seed_command": (
                "GPU=0 python scripts/66_reproduce_external_convnext_full_v094.py "
                "--data-project . "
                "--convnext-external training/checkpoints/external_convnext/convnext_best_map.pt "
                "--config configs/full_cv_pipeline.yaml "
                "--run-root runs/reproduce_external_convnext_full_v094 "
                "--deploy deploy/models_v094_external_global"
                if packaged_seed is not None else None
            ),
            "origin": origin,
            "provenance": ext,
            "external_training_source_audit": training_source_audit,
        },
        "full_training_reproducible_from_raw_data": True,
        "note": (
            "The team developer training source that creates convnext_best_map.pt is vendored under "
            "third_party/external_convnext_v5_training. Developer inference/export code is intentionally not vendored; "
            "the production v0.9.4 inference remains canonical."
        ),
    }
    (out / "TRAINING_REPRODUCIBILITY.json").write_text(json.dumps(training_repro, ensure_ascii=False, indent=2), encoding="utf-8")

    droot = out / "deploy" / "models_current"
    droot.mkdir(parents=True, exist_ok=True)
    slim = _sanitize_top_deployment(meta)
    (droot / "deployment.json").write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in meta["members"]:
        name = str(item["name"])
        src_member = Path(item["path"])
        if not src_member.is_absolute(): src_member = (dep / src_member).resolve()
        _copy_member_runtime(src_member, droot / "members" / name)

    req = out / "requirements-runtime-a5000.txt"
    non_exact = _exact_requirement_lines(req)
    if non_exact:
        raise RuntimeError(f"Non-exact runtime requirements: {non_exact}")
    train_req = out / "requirements-training.txt"
    train_non_exact = _exact_requirement_lines(train_req)
    if train_non_exact:
        raise RuntimeError(f"Non-exact training requirements: {train_non_exact}")
    ext_train_req = out / "requirements-training-external-convnext.txt"
    ext_train_non_exact = _exact_requirement_lines(ext_train_req)
    if ext_train_non_exact:
        raise RuntimeError(f"Non-exact external ConvNeXt reproduction requirements: {ext_train_non_exact}")

    weight_files = []
    forbidden = []
    total = 0
    for p in sorted(out.rglob("*")):
        if not p.is_file(): continue
        extn = p.suffix.lower()
        if extn in FORBIDDEN_FINAL_EXTS:
            forbidden.append(str(p.relative_to(out)))
        if extn in WEIGHT_EXTS:
            size = p.stat().st_size
            total += size
            weight_files.append({
                "path": str(p.relative_to(out)),
                "size_bytes": int(size),
                "sha256": sha256(p),
            })
    if forbidden:
        raise RuntimeError(f"Non-PT runtime artifacts are forbidden in final PT package: {forbidden}")
    if not any(Path(x["path"]).suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"} for x in weight_files):
        raise RuntimeError("Final PT package contains no PyTorch model weights")
    if total > MAX_WEIGHT_BYTES:
        raise RuntimeError(f"All packaged model weights exceed organizer 2GB limit: {total / 1024**3:.3f} GiB")

    (out / "WEIGHTS_SHA256SUMS.txt").write_text(
        "\n".join(f'{x["sha256"]}  {x["path"]}' for x in weight_files) + "\n", encoding="utf-8"
    )
    audit = {
        "schema": "vehicle-reid-v094-final-pt-package-audit-v1",
        "production_ensemble": EXPECTED,
        "fusion": "per_query_zscore",
        "runtime": "PyTorch CUDA",
        "recommended_precision": "fp16",
        "weight_files": weight_files,
        "total_weight_bytes": int(total),
        "total_weight_gib": float(total / 1024**3),
        "organizer_weight_limit_bytes": MAX_WEIGHT_BYTES,
        "under_2gb_limit": True,
        "onnx_files": [],
        "tensorrt_files": [],
        "exact_dependency_versions": True,
        "external_convnext_provenance": ext,
        "training_reproducibility": training_repro,
        "training_code_packaged": True,
        "external_convnext_seed_packaged": packaged_seed is not None,
        "external_convnext_seed": packaged_seed,
    }
    (out / "FINAL_PACKAGE_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"[OK] final PT-only package: {out}")


if __name__ == "__main__":
    main()
