#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MEMBERS = ["base_full", "direct_vit_full", "external_convnext_full", "mixed_refit_full"]
DEFAULT_WEIGHTS = [0.45, 0.15, 0.10, 0.30]
DEFAULT_MEMBER_COSTS = {"base_full": 1, "direct_vit_full": 1, "external_convnext_full": 1, "mixed_refit_full": 2}


def parse_named(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--member expects NAME=/path/to/deployment")
    name, value = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("empty member name")
    return name, Path(value).expanduser().resolve()


def select_from_metrics(path: Path, *, max_forward_cost: int | None = None, member_costs: dict[str, int] | None = None) -> dict:
    df = pd.read_csv(path)
    need = {"kind", "members", "weights", "fusion", "mAP@10_mean"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    e = df[df["kind"].astype(str).eq("ensemble_no_train")].copy()
    if e.empty:
        raise ValueError(f"{path}: no ensemble_no_train rows")
    costs = dict(DEFAULT_MEMBER_COSTS); costs.update(member_costs or {})
    if max_forward_cost is not None:
        def _cost(raw):
            names = str(raw).split("+")
            missing = [n for n in names if n not in costs]
            if missing:
                raise ValueError(f"No forward cost configured for members: {missing}")
            return sum(int(costs[n]) for n in names)
        e["forward_cost"] = e["members"].map(_cost)
        e = e[e["forward_cost"] <= int(max_forward_cost)].copy()
        if e.empty:
            raise ValueError(f"{path}: no ensemble_no_train rows fit max_forward_cost={max_forward_cost}")
    e["mAP@10_mean"] = pd.to_numeric(e["mAP@10_mean"], errors="coerce")
    sort_cols = [c for c in ["mAP@10_mean", "Rank-1_mean", "Rank-5_mean", "mAP_full_mean", "mINP_mean"] if c in e.columns]
    best = e.sort_values(sort_cols, ascending=False, kind="stable").iloc[0]
    members = str(best["members"]).split("+")
    weights = [float(x) for x in str(best["weights"]).split(",")]
    if len(members) != len(weights):
        raise ValueError(f"winner has mismatched members/weights: {best['members']} / {best['weights']}")
    if str(best["fusion"]) != "per_query_zscore":
        raise ValueError(f"winner fusion {best['fusion']!r} is not supported by deployment runtime")
    metrics = {}
    for c in ["mAP@10_mean", "mAP@10_std", "Rank-1_mean", "Rank-5_mean", "mAP_full_mean", "mINP_mean"]:
        if c in best and pd.notna(best[c]):
            metrics[c] = float(best[c])
    return {
        "variant": str(best.get("variant", "")),
        "members": members,
        "weights": weights,
        "fusion": str(best["fusion"]),
        "metrics": metrics,
        "source": str(path.resolve()),
        "forward_cost": int(best["forward_cost"]) if "forward_cost" in best and pd.notna(best["forward_cost"]) else None,
    }


def validate_deployment(path: Path) -> dict:
    mp = path / "deployment.json"
    if not mp.is_file():
        raise FileNotFoundError(mp)
    meta = json.loads(mp.read_text(encoding="utf-8"))
    mode = str(meta.get("mode", "single")).lower()
    if mode == "score_ensemble":
        raise ValueError(f"Nested score_ensemble is not supported: {path}")
    return meta


def copy_tree(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src, target_is_directory=True)
        return
    if mode == "copy":
        shutil.copytree(src, dst, symlinks=False)
        return
    if mode == "hardlink":
        def _copy(src_file, dst_file):
            try:
                os.link(src_file, dst_file)
            except OSError:
                shutil.copy2(src_file, dst_file)
        shutil.copytree(src, dst, copy_function=_copy, symlinks=False)
        return
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser(description="Build the selected v0.9.4 output-score ensemble deployment")
    ap.add_argument("--member", action="append", default=[], help="NAME=/deployment/path; repeat for every candidate")
    ap.add_argument("--metrics-summary", help="all_metrics_summary.csv; winner is selected by official mAP@10, then tie-breakers")
    ap.add_argument("--max-forward-cost", type=int, default=None, help="Optional speed budget. Default costs: single members=1, mixed_refit_full=2.")
    ap.add_argument("--member-cost", action="append", default=[], help="Override inference cost as NAME=INTEGER; repeat as needed")
    ap.add_argument("--out", default="deploy/models_current")
    ap.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    ap.add_argument("--candidate-member", default="base_full", help="Member whose calibrated candidates/refusal + embeddings.npy are retained")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    supplied = dict(parse_named(x) for x in a.member)
    if len(supplied) != len(a.member):
        raise SystemExit("Duplicate --member name")
    member_costs = {}
    for raw in a.member_cost:
        if "=" not in raw: raise SystemExit("--member-cost expects NAME=INTEGER")
        name, value = raw.split("=", 1); member_costs[name.strip()] = int(value)
        if member_costs[name.strip()] < 1: raise SystemExit("member cost must be >= 1")

    if a.metrics_summary:
        selection = select_from_metrics(Path(a.metrics_summary).expanduser().resolve(), max_forward_cost=a.max_forward_cost, member_costs=member_costs)
    else:
        selection = {
            "variant": "Z::" + "+".join(DEFAULT_MEMBERS) + "::" + ",".join(f"{x:.3f}" for x in DEFAULT_WEIGHTS),
            "members": list(DEFAULT_MEMBERS), "weights": list(DEFAULT_WEIGHTS),
            "fusion": "per_query_zscore", "metrics": {}, "source": None, "forward_cost": 5,
        }

    missing = [name for name in selection["members"] if name not in supplied]
    if missing:
        raise SystemExit(f"Winner needs member paths not supplied with --member: {missing}")
    if a.candidate_member not in selection["members"]:
        raise SystemExit(f"--candidate-member {a.candidate_member!r} is not in selected ensemble")

    out = Path(a.out).expanduser()
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    if out.exists():
        if not a.force:
            raise SystemExit(f"Output exists: {out}. Re-run with --force to replace it.")
        if out.is_symlink() or out.is_file(): out.unlink()
        else: shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    members_meta = []
    for name, weight in zip(selection["members"], selection["weights"]):
        src = supplied[name]
        source_meta = validate_deployment(src)
        dst = out / "members" / name
        copy_tree(src, dst, a.copy_mode)
        members_meta.append({
            "name": name,
            "path": str(Path("members") / name),
            "weight": float(weight),
            "source": str(src),
            "source_mode": str(source_meta.get("mode", "single")),
        })

    payload = {
        "schema": "vehicle-reid-v094-score-ensemble-v1",
        "mode": "score_ensemble",
        "fusion": selection["fusion"],
        "members": members_meta,
        "candidate_policy": {
            "mode": "inherit_member",
            "member": a.candidate_member,
            "note": "Shared-validation sweep selected ranking fusion only; refusal was not recalibrated for fused scores.",
        },
        "selection": {
            "variant": selection["variant"],
            "metrics": selection["metrics"],
            "metrics_summary": selection["source"],
            "selection_warning": "Weights selected on shared validation; hidden test remains untouched.",
        },
        "runtime": {
            "exact_ranking": True,
            "estimated_feature_forward_cost": selection.get("forward_cost"),
            "recommended_precision_nvidia": "fp16",
            "recommended_batch_sweep": [16, 24, 32, 48, 64],
        },
    }
    (out / "deployment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[OK] score ensemble deployment: {out}")


if __name__ == "__main__":
    main()
