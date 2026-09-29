#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vehicle_fingerprint.score_ensemble import read_score_ensemble_deployment

EXPECTED = {"base_full": 0.65, "external_convnext_full": 0.35}


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the fixed 0.65/0.35 production ensemble to ONNX Runtime models")
    ap.add_argument("--deployment-dir", default="deploy/models_current")
    ap.add_argument("--out", default="deploy/onnx_current")
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    dep = Path(a.deployment_dir).expanduser()
    if not dep.is_absolute():
        dep = (ROOT / dep).resolve()
    out = Path(a.out).expanduser()
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    meta = read_score_ensemble_deployment(dep)
    got = {m["name"]: float(m["weight"]) for m in meta["members"]}
    if set(got) != set(EXPECTED) or any(abs(got[k] - v) > 1e-9 for k, v in EXPECTED.items()):
        raise SystemExit(f"Production deployment must be exactly {EXPECTED}; got {got}")
    if str(meta.get("fusion")) != "per_query_zscore":
        raise SystemExit("Production fusion must be per_query_zscore")

    if out.exists() and a.force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for member in meta["members"]:
        src = Path(member["path"])
        sm = json.loads((src / "deployment.json").read_text(encoding="utf-8"))
        if str(sm.get("mode", "single")).lower() != "single":
            raise SystemExit(f"{member['name']} must be a single deployment")
        dst = out / "members" / member["name"]
        dst.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts/57_export_single_deployment_onnx_v094.py"),
            "--deployment-dir", str(src),
            "--out", str(dst),
            "--precision", a.precision,
            "--device", str(a.device),
            "--opset", str(a.opset),
        ]
        print("[RUN]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        records.append({
            "name": member["name"],
            "weight": float(member["weight"]),
            "deployment_mode": "single",
            "onnx_dir": str(Path("members") / member["name"]),
        })

    manifest = {
        "schema": "vehicle-reid-v094-production-onnx-v1",
        "precision": a.precision,
        "opset": a.opset,
        "deployment": str(dep),
        "provider": "CUDAExecutionProvider",
        "fusion": meta.get("fusion"),
        "shared_preprocess": True,
        "members": records,
    }
    (out / "onnx_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(dep / "deployment.json", out / "deployment.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[OK] ONNX export: {out}")


if __name__ == "__main__":
    main()
