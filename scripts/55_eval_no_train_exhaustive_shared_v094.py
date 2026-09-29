#!/usr/bin/env python3
from __future__ import annotations

"""
NO-TRAIN exhaustive evaluation of existing Vehicle ReID v0.9.4 models on the
fixed shared validation.

Inputs:
  --deployment NAME=/path/to/deploy_dir
      Exact full inference of an already built deployment. Single and ensemble
      deployment.json formats are supported.

  --checkpoint NAME=/path/to/checkpoint.pt
      Representation-only diagnostic of an existing checkpoint using z_global.
      Native V5 checkpoints are converted to project format in the output dir.

After singles are evaluated, the script builds NO-TRAIN output-level ensembles
from their FINAL ranking scores. Thus every deployment keeps its own
reranker/recipe before fusion; no reranker is applied to an incompatible feature
space.

No optimization/training/calibration is performed.
"""

import argparse
import importlib.util
import itertools
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.features import ensemble_feature_caches, extract_feature_cache, load_cache
from vehicle_fingerprint.official_validation import evaluate_recipe, generate_official_artifacts, run_official_script
from vehicle_fingerprint.v5_checkpoint import convert_v5_checkpoint


def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip())
    if not s:
        raise ValueError("empty name")
    return s


def parse_named(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("expected NAME=/path")
    n, p = raw.split("=", 1)
    return safe_name(n), Path(p).expanduser().resolve()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(root: Path, value, fallback=None, optional=False):
    if value in (None, "", False):
        if fallback is None:
            return None
        p = root / fallback
        if optional and not p.exists():
            return None
        value = fallback
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def load_official_module():
    path = ROOT / "official" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("official_eval_script", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


OFFICIAL = load_official_module()


def recipe_kwargs(recipe: dict, reranker: Path | None, device: str) -> dict:
    return {
        "base_alpha": float(recipe.get("base_alpha", 0.0)),
        "reranker_path": str(reranker) if reranker else None,
        "reranker_beta": float(recipe.get("reranker_beta", 0.0)),
        "rerank_topk": int(recipe.get("rerank_topk", 100)),
        "kreciprocal_lambda": float(recipe.get("kreciprocal_lambda", 0.0)),
        "kreciprocal_k": int(recipe.get("kreciprocal_k", 20)),
        "same_camera_filter": bool(recipe.get("same_camera_filter", False)),
        "device": device,
        "evaluator_path": "official/evaluate.py",
    }


def load_deployment(name: str, root: Path) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(root)
    mp = root / "deployment.json"
    meta = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
    mode = str(meta.get("mode", "single")).lower()

    reranker = resolve(root, meta.get("reranker"), "reranker.pt", optional=True)
    refusal = resolve(root, meta.get("refusal"), "refusal.json")
    recipe = resolve(root, meta.get("retrieval_recipe"), "retrieval_recipe.json")
    if not refusal or not refusal.is_file():
        raise FileNotFoundError(f"{name}: refusal missing: {refusal}")
    if not recipe or not recipe.is_file():
        raise FileNotFoundError(f"{name}: recipe missing: {recipe}")

    if mode == "ensemble":
        fallback_a = "reid_convnext.pt" if (root / "reid_convnext.pt").exists() else "reid_convnext_small.pt"
        fallback_b = "reid_vit.pt" if (root / "reid_vit.pt").exists() else "reid_vit_base.pt"
        a = resolve(root, meta.get("checkpoint_a"), fallback_a)
        b = resolve(root, meta.get("checkpoint_b"), fallback_b)
        w = meta.get("weight_a", meta.get("convnext_weight"))
        if w is None:
            raise RuntimeError(f"{name}: ensemble deployment has no weight_a/convnext_weight")
        ckpts = [Path(a), Path(b)]
        weight_a = float(w)
    else:
        ck = resolve(root, meta.get("checkpoint"), "reid.pt")
        ckpts = [Path(ck)]
        weight_a = None

    for p in ckpts:
        if not p.is_file():
            raise FileNotFoundError(f"{name}: checkpoint missing: {p}")
    if reranker is not None and not reranker.is_file():
        raise FileNotFoundError(f"{name}: reranker missing: {reranker}")

    return {
        "name": name,
        "kind": "deployment",
        "root": root,
        "mode": mode,
        "metadata": meta,
        "checkpoints": ckpts,
        "weight_a": weight_a,
        "reranker": reranker,
        "refusal": Path(refusal),
        "recipe": Path(recipe),
    }


def ensure_project_checkpoint(name: str, src: Path, out_root: Path) -> Path:
    import torch

    raw = torch.load(src, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model" in raw and "model_cfg" in raw:
        return src
    if isinstance(raw, dict) and "model_state" in raw and "model_name" in raw:
        out = out_root / "_converted_checkpoints" / f"{name}.project.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.is_file():
            print(f"[CONVERT] {src} -> {out}", flush=True)
            convert_v5_checkpoint(src, out)
        return out
    raise RuntimeError(
        f"{name}: unsupported checkpoint format {src}; expected project "
        "(model/model_cfg) or native V5 (model_state/model_name)"
    )


def load_raw_checkpoint(name: str, src: Path, out_root: Path) -> dict:
    if not src.is_file():
        raise FileNotFoundError(src)
    project = ensure_project_checkpoint(name, src, out_root)
    return {
        "name": name,
        "kind": "checkpoint",
        "mode": "raw_global",
        "root": src.parent,
        "metadata": {},
        "checkpoints": [project],
        "source_checkpoint": src,
        "weight_a": None,
        "reranker": None,
        "refusal": None,
        "recipe": None,
    }


def cache_ids(cache: dict) -> list[str]:
    for k in ("meta_image_id", "sample_id"):
        if k in cache:
            return cache[k].astype(str).tolist()
    raise KeyError("feature cache has no meta_image_id/sample_id")


def build_candidate_cache(cand: dict, manifest: Path, dst: Path, args) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if cand["kind"] == "checkpoint" or cand["mode"] != "ensemble":
        if args.resume and dst.is_file():
            return dst
        extract_feature_cache(
            manifest,
            cand["checkpoints"][0],
            dst,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch,
            workers=args.workers,
        )
        return dst

    a = dst.with_name(dst.stem + "_a.npz")
    b = dst.with_name(dst.stem + "_b.npz")
    if not (args.resume and a.is_file()):
        extract_feature_cache(
            manifest,
            cand["checkpoints"][0],
            a,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch,
            workers=args.workers,
        )
    if not (args.resume and b.is_file()):
        extract_feature_cache(
            manifest,
            cand["checkpoints"][1],
            b,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch,
            workers=args.workers,
        )
    if not (args.resume and dst.is_file()):
        ensemble_feature_caches(a, b, dst, weight_a=float(cand["weight_a"]))
    return dst


def dense_final_scores(ranked: dict, scores: dict, qids: list[str], gids: list[str]) -> np.ndarray:
    gidx = {str(g): i for i, g in enumerate(gids)}
    out = np.full((len(qids), len(gids)), np.nan, np.float32)
    for qi, qid in enumerate(qids):
        rr = ranked.get(str(qid), [])
        ss = np.asarray(scores.get(str(qid), []), dtype=np.float32).reshape(-1)
        n = min(len(rr), len(ss))
        for gid, s in zip(rr[:n], ss[:n]):
            j = gidx.get(str(gid))
            if j is not None:
                out[qi, j] = float(s)
        finite = np.isfinite(out[qi])
        if finite.any():
            lo = float(np.min(out[qi, finite]))
            spread = float(np.std(out[qi, finite]))
            out[qi, ~finite] = lo - max(spread, 1e-3) * 10.0
        else:
            out[qi, :] = -1e6
    return out


def zscore_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-6)


def ranks_from_matrix(x: np.ndarray) -> np.ndarray:
    order = np.argsort(-x, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int32)
    row = np.arange(x.shape[0])[:, None]
    ranks[row, order] = np.arange(1, x.shape[1] + 1, dtype=np.int32)[None, :]
    return ranks


def ranked_dict_from_matrix(x: np.ndarray, qids: list[str], gids: list[str]) -> dict:
    order = np.argsort(-x, axis=1, kind="stable")
    return {str(qid): [str(gids[j]) for j in order[i]] for i, qid in enumerate(qids)}


def full_metrics_from_ranked(query: pd.DataFrame, gallery: pd.DataFrame, ranked: dict) -> dict:
    gal_vid = {str(k): str(v) for k, v in gallery.vehicle_id.to_dict().items()}
    gal_cam = {str(k): str(v) for k, v in gallery.camera_id.to_dict().items()}
    aps, inps = [], []
    for qid, row in query.iterrows():
        qid = str(qid)
        qvid, qcam = str(row.vehicle_id), str(row.camera_id)
        seq = ranked.get(qid, [])
        clean = [
            str(g)
            for g in seq
            if str(g) in gal_vid and not (gal_vid[str(g)] == qvid and gal_cam[str(g)] == qcam)
        ]
        rel = np.asarray([gal_vid[g] == qvid for g in clean], dtype=bool)
        n_pos = int(rel.sum())
        if n_pos == 0:
            continue
        cum = np.cumsum(rel)
        prec = cum / (np.arange(len(rel)) + 1)
        aps.append(float((prec * rel).sum() / n_pos))
        hardest = int(np.max(np.flatnonzero(rel))) + 1
        inps.append(float(n_pos / hardest))
    return {
        "mAP_full": float(np.mean(aps)) if aps else 0.0,
        "mINP": float(np.mean(inps)) if inps else 0.0,
        "n_scored": len(aps),
    }


def ranking_metrics_from_matrix(x: np.ndarray, qids, gids, gt: Path) -> dict:
    query, gallery = OFFICIAL.load_gt(str(gt))
    query.index = query.index.astype(str)
    gallery.index = gallery.index.astype(str)
    ranked = ranked_dict_from_matrix(x, qids, gids)
    rm = OFFICIAL.ranking_metrics(query, gallery, ranked, top_k=10)
    fm = full_metrics_from_ranked(query, gallery, ranked)
    return {**rm, **fm}


def simplex_weights(k: int, step: float):
    units = int(round(1.0 / step))
    if not math.isclose(units * step, 1.0, abs_tol=1e-9):
        raise ValueError("--weight-step must divide 1.0 exactly: 0.5,0.25,0.2,0.1,0.05")

    def comp(total, parts, prefix=()):
        if parts == 1:
            if total >= 1:
                yield prefix + (total,)
            return
        for v in range(1, total - parts + 2):
            yield from comp(total - v, parts - 1, prefix + (v,))

    yield from (tuple(v / units for v in ints) for ints in comp(units, k))


def metric_row(label, fold, m, *, kind, members="", weights="", fusion=""):
    def f(k):
        v = m.get(k, np.nan)
        return float(v) if v is not None else np.nan

    return {
        "variant": label,
        "kind": kind,
        "fold": int(fold),
        "members": members,
        "weights": weights,
        "fusion": fusion,
        "mAP@10": f("mAP@10"),
        "Rank-1": f("Rank-1"),
        "Rank-5": f("Rank-5"),
        "mAP_full": f("mAP_full"),
        "mINP": f("mINP"),
        "Precision": f("Precision"),
        "Recall": f("Recall"),
        "F1": f("F1"),
        "TNR": f("TNR"),
        "PR-AUC": f("PR-AUC"),
    }


def aggregate(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    metrics = ["mAP@10", "Rank-1", "Rank-5", "mAP_full", "mINP", "Precision", "Recall", "F1", "TNR", "PR-AUC"]
    for key, g in df.groupby(["variant", "kind", "members", "weights", "fusion"], dropna=False, sort=False):
        r = dict(zip(["variant", "kind", "members", "weights", "fusion"], key))
        r["folds"] = int(g["fold"].nunique())
        for m in metrics:
            v = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(float)
            r[m + "_mean"] = float(v.mean()) if len(v) else np.nan
            r[m + "_std"] = float(v.std()) if len(v) else np.nan
        out.append(r)
    ans = pd.DataFrame(out)
    return ans.sort_values(
        ["mAP@10_mean", "Rank-1_mean", "Rank-5_mean", "mAP_full_mean", "mINP_mean"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="NO-TRAIN exhaustive shared-validation benchmark")
    ap.add_argument("--config", default="configs/full_cv_pipeline.yaml")
    ap.add_argument("--cv-root", default=None)
    ap.add_argument("--deployment", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PT")
    ap.add_argument("--out", default="artifacts/no_train_exhaustive_shared")
    ap.add_argument("--device", default="0")
    ap.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--weight-step", type=float, default=0.1)
    ap.add_argument("--max-ensemble-size", type=int, default=4)
    ap.add_argument("--rrf-k", type=float, default=60.0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip-split-verification", action="store_true")
    args = ap.parse_args()
    args.resume = not args.no_resume

    cfgp = Path(args.config).expanduser()
    if not cfgp.is_absolute():
        cfgp = ROOT / cfgp
    cfg = read_yaml(cfgp)
    cv_root = Path(args.cv_root).expanduser().resolve() if args.cv_root else Path(cfg["paths"]["cv"]).expanduser()
    if not cv_root.is_absolute():
        cv_root = (ROOT / cv_root).resolve()

    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_split_verification:
        spec = Path(cfg["cv"]["shared_validation_spec"])
        if not spec.is_absolute():
            spec = ROOT / spec
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "25b_verify_shared_validation.py"),
            "--cv-root",
            str(cv_root),
            "--spec",
            str(spec),
        ]
        print("[VERIFY]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)

    candidates = []
    for raw in args.deployment:
        candidates.append(load_deployment(*parse_named(raw)))
    for raw in args.checkpoint:
        n, p = parse_named(raw)
        candidates.append(load_raw_checkpoint(n, p, out))
    if not candidates:
        raise SystemExit("Provide at least one --deployment or --checkpoint")
    names = [x["name"] for x in candidates]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate candidate names: {names}")

    shared = cv_root / "shared_validation"
    folds = sorted(
        [p for p in shared.glob("fold_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not folds:
        raise FileNotFoundError(f"No shared_validation/fold_* under {cv_root}")

    all_rows = []
    for fold, fd in enumerate(folds):
        print(f"\n{'=' * 100}\nFOLD {fold}: {fd}\n{'=' * 100}")
        qcsv, gcsv, gt = fd / "query.csv", fd / "gallery.csv", fd / "ground_truth.csv"
        qdf = pd.read_csv(qcsv, dtype={"image_id": str})
        gdf = pd.read_csv(gcsv, dtype={"image_id": str})
        qids = qdf["image_id"].astype(str).tolist()
        gids = gdf["image_id"].astype(str).tolist()

        final_mats = {}
        final_ranks = {}

        for cand in candidates:
            name = cand["name"]
            cd = out / "candidates" / name / f"fold_{fold}"
            cd.mkdir(parents=True, exist_ok=True)
            qc = build_candidate_cache(cand, qcsv, cd / "query.npz", args)
            gc = build_candidate_cache(cand, gcsv, cd / "gallery.npz", args)
            qcache, gcache = load_cache(qc), load_cache(gc)
            if cache_ids(qcache) != qids or cache_ids(gcache) != gids:
                raise RuntimeError(f"{name}/fold{fold}: cache row order mismatch")

            device = "cuda" if str(args.device) not in {"cpu", "-1"} else "cpu"
            if cand["kind"] == "deployment":
                recipe = json.loads(cand["recipe"].read_text(encoding="utf-8"))
                refusal = json.loads(cand["refusal"].read_text(encoding="utf-8"))
                report_path = cd / "official_report.json"
                if args.resume and report_path.is_file():
                    rep_official = json.loads(report_path.read_text(encoding="utf-8"))
                else:
                    generate_official_artifacts(
                        qc,
                        gc,
                        gt,
                        cd,
                        recipe,
                        refusal,
                        reranker_path=cand["reranker"],
                        device=device,
                        evaluator_path="official/evaluate.py",
                    )
                    rep_official = run_official_script(
                        gt,
                        cd / "submission.csv",
                        candidates=cd / "candidates.csv",
                        embeddings=cd / "embeddings.npy",
                        query_csv=qcsv,
                        gallery_csv=gcsv,
                        json_out=report_path,
                        evaluator_path="official/evaluate.py",
                    )
                rep_rank, ranked, scores, *_ = evaluate_recipe(
                    qc,
                    gc,
                    gt,
                    **recipe_kwargs(recipe, cand["reranker"], device),
                )
                m = {}
                m.update(rep_official.get("ranking", {}))
                m.update(rep_official.get("full_ranking", {}))
                m.update(rep_official.get("candidates", {}))
            else:
                rep_rank, ranked, scores, *_ = evaluate_recipe(
                    qc,
                    gc,
                    gt,
                    base_alpha=0.0,
                    device=device,
                    evaluator_path="official/evaluate.py",
                )
                m = {}
                m.update(rep_rank.get("ranking", {}))
                m.update(rep_rank.get("full_ranking", {}))

            final_mats[name] = dense_final_scores(ranked, scores, qids, gids)
            final_ranks[name] = ranks_from_matrix(final_mats[name])
            all_rows.append(
                metric_row(name, fold, m, kind=cand["kind"], members=name, weights="1.0", fusion="native")
            )
            print(
                f"[{name}] mAP@10={float(m.get('mAP@10', 0)):.6f} "
                f"R1={float(m.get('Rank-1', 0)):.6f} full={float(m.get('mAP_full', 0)):.6f}"
            )

        max_k = min(int(args.max_ensemble_size), len(names))
        z = {n: zscore_rows(final_mats[n]) for n in names}
        for k in range(2, max_k + 1):
            for members in itertools.combinations(names, k):
                member_str = "+".join(members)

                rrf = np.zeros_like(final_mats[members[0]], dtype=np.float32)
                for n in members:
                    rrf += 1.0 / (float(args.rrf_k) + final_ranks[n].astype(np.float32))
                mm = ranking_metrics_from_matrix(rrf, qids, gids, gt)
                all_rows.append(
                    metric_row(
                        f"RRF::{member_str}",
                        fold,
                        mm,
                        kind="ensemble_no_train",
                        members=member_str,
                        weights="equal",
                        fusion=f"rrf_k{args.rrf_k:g}",
                    )
                )

                for ws in simplex_weights(k, float(args.weight_step)):
                    fused = np.zeros_like(final_mats[members[0]], dtype=np.float32)
                    for n, w in zip(members, ws):
                        fused += float(w) * z[n]
                    mm = ranking_metrics_from_matrix(fused, qids, gids, gt)
                    wstr = ",".join(f"{w:.3f}" for w in ws)
                    all_rows.append(
                        metric_row(
                            f"Z::{member_str}::{wstr}",
                            fold,
                            mm,
                            kind="ensemble_no_train",
                            members=member_str,
                            weights=wstr,
                            fusion="per_query_zscore",
                        )
                    )

    per_fold = pd.DataFrame(all_rows)
    per_fold.to_csv(out / "all_metrics_by_fold.csv", index=False)
    summary = aggregate(all_rows)
    summary.to_csv(out / "all_metrics_summary.csv", index=False)
    summary[~summary["kind"].eq("ensemble_no_train")].to_csv(out / "singles_summary.csv", index=False)
    summary[summary["kind"].eq("ensemble_no_train")].to_csv(out / "ensembles_summary.csv", index=False)

    payload = {
        "evaluation_only": True,
        "training_performed": False,
        "cv_root": str(cv_root),
        "candidate_names": names,
        "weight_step": float(args.weight_step),
        "max_ensemble_size": int(args.max_ensemble_size),
        "warning": (
            "If ensemble weights are chosen from this shared validation, the selected shared score "
            "is a model-selection score, not an untouched final estimate. Hidden test remains untouched."
        ),
        "top20": summary.head(20).replace({np.nan: None}).to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [
        "variant",
        "kind",
        "members",
        "weights",
        "fusion",
        "mAP@10_mean",
        "mAP@10_std",
        "Rank-1_mean",
        "Rank-5_mean",
        "mAP_full_mean",
        "mINP_mean",
        "F1_mean",
        "TNR_mean",
    ]
    print("\n=== TOP NO-TRAIN RESULTS ===")
    print(summary[cols].head(40).to_string(index=False))
    print(
        f"\nSaved:\n  {out / 'singles_summary.csv'}\n  {out / 'ensembles_summary.csv'}\n  {out / 'all_metrics_summary.csv'}"
    )


if __name__ == "__main__":
    main()
