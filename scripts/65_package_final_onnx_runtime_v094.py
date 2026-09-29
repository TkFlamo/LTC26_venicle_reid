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
FORBIDDEN_FINAL_EXTS = {".engine", ".plan", ".trt"}
MAX_WEIGHT_BYTES = 2 * 1024**3


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


def _copy_runtime_member_metadata(src_member: Path, dst_member: Path) -> None:
    dst_member.mkdir(parents=True, exist_ok=True)
    required = ["deployment.json", "retrieval_recipe.json"]
    optional = ["refusal.json"]
    for name in required:
        p = src_member / name
        if not p.is_file():
            raise FileNotFoundError(p)
        shutil.copy2(p, dst_member / name)
    for name in optional:
        p = src_member / name
        if p.is_file():
            shutil.copy2(p, dst_member / name)


def _sanitize_top_deployment(meta: dict) -> dict:
    out = dict(meta)
    members = []
    for item in meta["members"]:
        members.append({
            "name": str(item["name"]),
            "path": str(Path("members") / str(item["name"])),
            "weight": float(item["weight"]),
            "source_mode": "single",
        })
    out["members"] = members
    out["final_runtime"] = {
        "feature_runtime": "onnxruntime_cuda_only",
        "pytorch_checkpoints_in_final_package": False,
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Create final ONNX-only hackathon runtime package (<2GB inference weights)")
    ap.add_argument("--deployment-dir", default="deploy/models_current")
    ap.add_argument("--onnx-dir", default="deploy/onnx_current")
    ap.add_argument("--out", default="dist/falcon_reid_v094_onnx")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-unresolved-provenance", action="store_true")
    a = ap.parse_args()

    dep = Path(a.deployment_dir).expanduser()
    if not dep.is_absolute(): dep = (ROOT / dep).resolve()
    onnx = Path(a.onnx_dir).expanduser()
    if not onnx.is_absolute(): onnx = (ROOT / onnx).resolve()
    out = Path(a.out).expanduser()
    if not out.is_absolute(): out = (ROOT / out).resolve()

    meta_path = dep / "deployment.json"
    onnx_manifest_path = onnx / "onnx_manifest.json"
    if not meta_path.is_file(): raise FileNotFoundError(meta_path)
    if not onnx_manifest_path.is_file(): raise FileNotFoundError(onnx_manifest_path)
    meta = _read_json(meta_path)
    _validate_production(meta)
    om = _read_json(onnx_manifest_path)
    if str(om.get("schema")) != "vehicle-reid-v094-production-onnx-v1":
        raise RuntimeError(f"Unexpected ONNX manifest schema: {om.get('schema')}")
    if str(om.get("precision")) not in {"fp16", "fp32"}:
        raise RuntimeError(f"Unsupported final ONNX precision: {om.get('precision')}")

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

    # Code/runtime. Keep source tree so the package remains auditable; no training weights are copied.
    for d in ("src", "scripts", "official"):
        shutil.copytree(ROOT / d, out / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in (
        "pyproject.toml", "requirements-runtime-a5000.txt", "run_hackathon.py",
        "HACKATHON_COMPLIANCE.md", "ONNX_RUNTIME_V094.md", "PRODUCTION_MODEL_V094.md",
    ):
        p = ROOT / name
        if p.is_file(): shutil.copy2(p, out / name)
    shutil.copy2(ROOT / "Dockerfile.runtime", out / "Dockerfile")

    # Slim deployment metadata: required by the inference contract, but no PT/PTH checkpoints.
    droot = out / "deploy" / "models_current"
    droot.mkdir(parents=True, exist_ok=True)
    slim = _sanitize_top_deployment(meta)
    (droot / "deployment.json").write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in meta["members"]:
        name = str(item["name"])
        src_member = Path(item["path"])
        if not src_member.is_absolute(): src_member = (dep / src_member).resolve()
        _copy_runtime_member_metadata(src_member, droot / "members" / name)

    # ONNX is the only inference-weight representation shipped to the organizers.
    shutil.copytree(onnx, out / "deploy" / "onnx_current")

    # Reproducibility/audit.
    req = out / "requirements-runtime-a5000.txt"
    non_exact = _exact_requirement_lines(req)
    if non_exact:
        raise RuntimeError(f"Non-exact runtime requirements: {non_exact}")

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
        raise RuntimeError(f"TensorRT artifacts are forbidden in final ONNX package: {forbidden}")
    pt_like = [x["path"] for x in weight_files if Path(x["path"]).suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}]
    if pt_like:
        raise RuntimeError(f"Final ONNX package unexpectedly contains PyTorch weights: {pt_like}")
    if total > MAX_WEIGHT_BYTES:
        raise RuntimeError(f"Inference weights exceed organizer 2GB limit: {total / 1024**3:.3f} GiB")

    sums = "\n".join(f'{x["sha256"]}  {x["path"]}' for x in weight_files) + "\n"
    (out / "WEIGHTS_SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    audit = {
        "schema": "vehicle-reid-v094-final-package-audit-v1",
        "production_ensemble": EXPECTED,
        "fusion": "per_query_zscore",
        "runtime": "onnxruntime-gpu / CUDAExecutionProvider",
        "onnx_precision": om.get("precision"),
        "weight_files": weight_files,
        "total_weight_bytes": int(total),
        "total_weight_gib": float(total / 1024**3),
        "organizer_weight_limit_bytes": MAX_WEIGHT_BYTES,
        "under_2gb_limit": True,
        "pytorch_weight_files": [],
        "tensorrt_files": [],
        "exact_dependency_versions": True,
        "external_convnext_provenance": ext,
    }
    (out / "FINAL_PACKAGE_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"[OK] final ONNX-only package: {out}")


if __name__ == "__main__":
    main()
