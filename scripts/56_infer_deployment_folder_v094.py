#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")


def _local_env() -> dict[str, str]:
    """Pin child processes to this dist's src before any inherited PYTHONPATH."""
    env = os.environ.copy()
    src = str((ROOT / "src").resolve())
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + old if old else "")
    return env


def _resolve_image(images: Path, image_id: str) -> Path | None:
    s = str(image_id)
    p = images / s
    if p.is_file():
        return p
    stem = Path(s).stem if Path(s).suffix else s
    for ext in IMAGE_EXTS:
        q = images / f"{stem}{ext}"
        if q.is_file():
            return q
    return None


def _validate_csv(path: Path, images: Path, label: str) -> int:
    if not path.is_file():
        raise SystemExit(f"Missing {label} CSV: {path}")
    df = pd.read_csv(path, dtype={"image_id": str})
    need = {"image_id", "x", "y", "w", "h"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"{path}: missing columns {sorted(miss)}")
    if df["image_id"].isna().any() or (df["image_id"].astype(str).str.len() == 0).any():
        raise SystemExit(f"{path}: empty image_id found")
    if df["image_id"].astype(str).duplicated().any():
        bad = df.loc[df["image_id"].astype(str).duplicated(False), "image_id"].astype(str).head(10).tolist()
        raise SystemExit(f"{path}: image_id must be unique; examples={bad}")
    for c in ("x", "y", "w", "h"):
        df[c] = pd.to_numeric(df[c], errors="raise")
    if (df["w"] <= 0).any() or (df["h"] <= 0).any():
        raise SystemExit(f"{path}: non-positive bbox found")
    missing = []
    for image_id in df["image_id"].astype(str):
        if _resolve_image(images, image_id) is None:
            missing.append(image_id)
            if len(missing) >= 20:
                break
    if missing:
        raise SystemExit(f"{label}: images not found in {images}; first missing IDs: {missing}")
    return len(df)


def _has_option(script: Path, option: str) -> bool:
    try:
        p = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT, env=_local_env(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        return option in (p.stdout or "")
    except Exception:
        return False


def _rel_or_abs(dep: Path, value: str | None, fallback: str | None = None) -> Path | None:
    value = value or fallback
    if not value:
        return None
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = dep / p
    return p.resolve()


def _select_score_ensemble_member(dep: Path, meta: dict, member_name: str) -> tuple[Path, dict]:
    """Resolve one score-ensemble member as a normal single deployment.

    The ensemble metadata remains untouched on disk.  This is a runtime-only
    switch so external_convnext_full stays packaged and reproducible while the
    default production path can execute only base_full.
    """
    members = {str(m.get("name")): m for m in meta.get("members", [])}
    if member_name not in members:
        raise SystemExit(
            f"Requested member {member_name!r} not found in {dep / 'deployment.json'}; "
            f"available={sorted(members)}"
        )
    spec = members[member_name]
    member_dep = Path(str(spec.get("path", ""))).expanduser()
    if not member_dep.is_absolute():
        member_dep = dep / member_dep
    member_dep = member_dep.resolve()

    mp = member_dep / "deployment.json"
    if mp.is_file():
        member_meta = json.loads(mp.read_text(encoding="utf-8"))
    else:
        # Compatibility fallback for slimmed packages.  The normal final package
        # contains member deployment.json, but canonical filenames are stable.
        member_meta = {
            "mode": "single",
            "checkpoint": "reid.pt",
            "reranker": "reranker.pt",
            "refusal": "refusal.json",
            "retrieval_recipe": "retrieval_recipe.json",
        }
    member_meta["mode"] = "single"
    return member_dep, member_meta


def main():
    ap = argparse.ArgumentParser(
        description="Run PT v0.9.4 deployment on input_dir/{test_query.csv,test_gallery.csv,images/}. "
                    "Default runtime uses base_full only; the packaged ConvNeXt member is left intact."
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--deployment-dir", default="deploy/models_current")
    ap.add_argument("--out", default="outputs/test_submission")
    ap.add_argument("--query-name", default="test_query.csv")
    ap.add_argument("--gallery-name", default="test_gallery.csv")
    ap.add_argument("--images-name", default="images")
    ap.add_argument("--device", default="0")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--member", choices=["base_full", "external_convnext_full", "ensemble"], default="base_full",
        help="Runtime member. Default: base_full. 'ensemble' restores the previous two-model score ensemble.",
    )
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()

    inp = Path(a.input_dir).expanduser().resolve()
    dep = Path(a.deployment_dir).expanduser()
    if not dep.is_absolute():
        dep = (ROOT / dep).resolve()
    out = Path(a.out).expanduser()
    if not out.is_absolute():
        out = (ROOT / out).resolve()

    qcsv = inp / a.query_name
    gcsv = inp / a.gallery_name
    images = inp / a.images_name
    if not images.is_dir():
        raise SystemExit(f"Missing images directory: {images}")

    nq = _validate_csv(qcsv, images, "query")
    ng = _validate_csv(gcsv, images, "gallery")
    overlap = set(pd.read_csv(qcsv, dtype={"image_id": str}).image_id.astype(str)) & set(
        pd.read_csv(gcsv, dtype={"image_id": str}).image_id.astype(str)
    )

    mp = dep / "deployment.json"
    if not mp.is_file():
        raise SystemExit(f"Missing deployment.json: {mp}")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    original_mode = str(meta.get("mode", "single")).lower()
    selected_member = a.member

    # Runtime-only member selection. Do not mutate/copy deployment metadata.
    if original_mode == "score_ensemble" and selected_member != "ensemble":
        dep, meta = _select_score_ensemble_member(dep, meta, selected_member)
        mode = "single"
    else:
        mode = original_mode
        if selected_member == "ensemble" and mode != "score_ensemble":
            raise SystemExit("--member ensemble requires a score_ensemble top-level deployment")

    print(json.dumps({
        "input_dir": str(inp),
        "query_rows": nq,
        "gallery_rows": ng,
        "query_gallery_image_id_overlap": len(overlap),
        "deployment_dir": str(dep),
        "packaged_deployment_mode": original_mode,
        "runtime_mode": mode,
        "runtime_member": selected_member,
        "runtime": "pytorch",
        "precision": a.precision,
        "out": str(out),
    }, ensure_ascii=False, indent=2))

    if a.validate_only:
        print("[OK] input/deployment validation passed")
        return

    if mode == "score_ensemble":
        script = ROOT / "scripts" / "61_infer_score_ensemble_folder_v094.py"
        cmd = [
            sys.executable, str(script),
            "--input-dir", str(inp),
            "--deployment-dir", str(dep),
            "--out", str(out),
            "--device", str(a.device),
            "--precision", str(a.precision),
            "--batch", str(a.batch),
            "--workers", str(a.workers),
        ]
    elif mode == "ensemble":
        script = ROOT / "scripts" / "22_infer_ensemble_from_csv.py"
        cmd = [
            sys.executable, str(script),
            "--query-csv", str(qcsv),
            "--query-images", str(images),
            "--gallery-csv", str(gcsv),
            "--gallery-images", str(images),
            "--out", str(out),
            "--device", str(a.device),
        ]
        if _has_option(script, "--deployment-dir"):
            cmd += ["--deployment-dir", str(dep)]
        else:
            ca = _rel_or_abs(dep, meta.get("checkpoint_a"), "reid_convnext.pt")
            cb = _rel_or_abs(dep, meta.get("checkpoint_b"), "reid_vit.pt")
            w = meta.get("weight_a", meta.get("convnext_weight"))
            if not ca or not ca.is_file() or not cb or not cb.is_file() or w is None:
                raise SystemExit("Incomplete ensemble deployment metadata/files")
            cmd += ["--checkpoint-a", str(ca), "--checkpoint-b", str(cb), "--weight-a", str(float(w))]
            rr = _rel_or_abs(dep, meta.get("reranker"), "reranker.pt")
            rf = _rel_or_abs(dep, meta.get("refusal"), "refusal.json")
            rc = _rel_or_abs(dep, meta.get("retrieval_recipe"), "retrieval_recipe.json")
            if rr and rr.is_file(): cmd += ["--reranker", str(rr)]
            if rf and rf.is_file(): cmd += ["--refusal", str(rf)]
            if rc and rc.is_file(): cmd += ["--recipe", str(rc)]
    else:
        script = ROOT / "scripts" / "21_infer_from_csv.py"
        ck = _rel_or_abs(dep, meta.get("checkpoint"), "reid.pt")
        rr = _rel_or_abs(dep, meta.get("reranker"), "reranker.pt")
        rf = _rel_or_abs(dep, meta.get("refusal"), "refusal.json")
        rc = _rel_or_abs(dep, meta.get("retrieval_recipe"), "retrieval_recipe.json")
        required = {"checkpoint": ck, "refusal": rf, "recipe": rc}
        missing = [f"{k}={v}" for k, v in required.items() if v is None or not v.is_file()]
        if missing:
            raise SystemExit("Incomplete single deployment: " + ", ".join(missing))
        cmd = [
            sys.executable, str(script),
            "--query-csv", str(qcsv),
            "--query-images", str(images),
            "--gallery-csv", str(gcsv),
            "--gallery-images", str(images),
            "--checkpoint", str(ck),
            "--refusal", str(rf),
            "--recipe", str(rc),
            "--out", str(out),
            "--device", str(a.device),
        ]
        if rr and rr.is_file():
            cmd += ["--reranker", str(rr)]

    if mode != "score_ensemble":
        for opt, val in (("--precision", a.precision), ("--batch", a.batch), ("--workers", a.workers)):
            if _has_option(script, opt):
                cmd += [opt, str(val)]

    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=_local_env(), check=True)

    expected = ["submission.csv", "candidates.csv", "embeddings.npy"]
    missing_out = [x for x in expected if not (out / x).is_file()]
    if missing_out:
        raise SystemExit(f"Inference finished but expected outputs are missing: {missing_out}")

    validator = ROOT / "scripts" / "64_validate_hackathon_io_v094.py"
    validation_json = out / "hackathon_io_validation.json"
    vcmd = [
        sys.executable, str(validator),
        "--input-dir", str(inp),
        "--output-dir", str(out),
        "--json-out", str(validation_json),
    ]
    print("[VALIDATE]", " ".join(vcmd), flush=True)
    subprocess.run(vcmd, cwd=ROOT, env=_local_env(), check=True)
    print(f"[OK] complete organizer-compatible PT inference outputs: {out}")


if __name__ == "__main__":
    main()
