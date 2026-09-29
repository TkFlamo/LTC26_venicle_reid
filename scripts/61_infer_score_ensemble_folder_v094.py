#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.data.prepare import prepare_hackathon_dataset
from vehicle_fingerprint.official_eval import normalize, write_candidates, write_embeddings, write_submission
from vehicle_fingerprint.official_validation import rank_caches
from vehicle_fingerprint.refusal import refusal_probability
from vehicle_fingerprint.runtime_pt import extract_feature_caches_loaded_pt_timed, load_multi_pt_models
from vehicle_fingerprint.score_ensemble import (
    dense_final_scores,
    fuse_score_matrices,
    ranked_dict_from_matrix,
    read_score_ensemble_deployment,
    weighted_concat_embeddings,
)

EXPECTED_WEIGHTS = {"base_full": 0.65, "external_convnext_full": 0.35}


def resolve(root: Path, value, fallback=None, optional=False) -> Path | None:
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


def cache_ids(cache: dict) -> list[str]:
    key = "meta_image_id" if "meta_image_id" in cache else "sample_id"
    return cache[key].astype(str).tolist()


def inherited_candidates(member_result: dict, qids: list[str], gids: list[str]):
    refusal = member_result["refusal"]
    ranked = member_result["ranked"]
    scores = member_result["scores"]
    q = member_result["qcache"]
    g = member_result["gcache"]
    if refusal is None:
        return []
    qglob = normalize(q["z_global"])
    gglob = normalize(g["z_global"])
    cache_qids = cache_ids(q)
    cache_gids = cache_ids(g)
    gidx = {str(x): i for i, x in enumerate(cache_gids)}
    qmap = {str(a): str(b) for a, b in zip(cache_qids, qids)}
    gmap = {str(a): str(b) for a, b in zip(cache_gids, gids)}
    rows = []
    for qi, cqid in enumerate(cache_qids):
        order = ranked.get(str(cqid), [])
        ss = np.asarray(scores.get(str(cqid), []), np.float32)
        if not order or len(ss) == 0:
            continue
        gi = gidx[str(order[0])]
        s1 = float(ss[0])
        s2 = float(ss[1]) if len(ss) > 1 else 0.0
        top5 = float(np.mean(ss[:min(5, len(ss))]))
        glob = float(qglob[qi] @ gglob[gi])
        X = np.asarray([[s1, s1 - s2, glob, s1 - top5]], np.float32)
        prob = float(refusal_probability(refusal, X)[0])
        if prob >= float(refusal["threshold"]):
            rows.append((
                qmap.get(str(cqid), str(cqid)),
                gmap.get(str(order[0]), str(order[0])),
                prob,
            ))
    return rows


def _validate_production(meta: dict) -> None:
    got = {m["name"]: float(m["weight"]) for m in meta["members"]}
    if set(got) != set(EXPECTED_WEIGHTS):
        raise RuntimeError(f"Expected production members {list(EXPECTED_WEIGHTS)}, got {list(got)}")
    for name, weight in EXPECTED_WEIGHTS.items():
        if abs(got[name] - weight) > 1e-9:
            raise RuntimeError(f"Expected {name} weight={weight}, got {got[name]}")
    if str(meta.get("fusion")) != "per_query_zscore":
        raise RuntimeError(f"Expected per_query_zscore fusion, got {meta.get('fusion')}")
    for m in meta["members"]:
        sm = json.loads((Path(m["path"]) / "deployment.json").read_text(encoding="utf-8"))
        if str(sm.get("mode", "single")).lower() != "single":
            raise RuntimeError(f"Production member {m['name']} must be mode=single")


def _torch_device_arg(device: str) -> str:
    s = str(device)
    return f"cuda:{s}" if s.isdigit() else s


def _rank_member(name: str, member: dict, qcache: dict, gcache: dict, reranker_device: str) -> dict:
    dep = Path(member["path"])
    meta = json.loads((dep / "deployment.json").read_text(encoding="utf-8"))
    recipe_path = resolve(dep, meta.get("retrieval_recipe"), "retrieval_recipe.json")
    refusal_path = resolve(dep, meta.get("refusal"), "refusal.json", optional=True)
    reranker_path = resolve(dep, meta.get("reranker"), "reranker.pt", optional=True)
    if recipe_path is None or not recipe_path.is_file():
        raise FileNotFoundError(recipe_path)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    refusal = json.loads(refusal_path.read_text(encoding="utf-8")) if refusal_path else None
    if float(recipe.get("reranker_beta", 0.0)) > 0 and (reranker_path is None or not reranker_path.is_file()):
        raise FileNotFoundError(f"{name}: retrieval recipe uses reranker_beta>0 but reranker.pt is missing")

    t = time.perf_counter()
    ranked, scores, qe, ge, _, _ = rank_caches(
        qcache,
        gcache,
        base_alpha=float(recipe.get("base_alpha", 0.0)),
        reranker_path=reranker_path,
        reranker_beta=float(recipe.get("reranker_beta", 0.0)),
        rerank_topk=int(recipe.get("rerank_topk", 100)),
        kreciprocal_lambda=float(recipe.get("kreciprocal_lambda", 0.0)),
        kreciprocal_k=int(recipe.get("kreciprocal_k", 20)),
        same_camera_filter=bool(recipe.get("same_camera_filter", False)),
        device=reranker_device,
    )
    retrieval_s = time.perf_counter() - t
    return {
        "qcache": qcache,
        "gcache": gcache,
        "ranked": ranked,
        "scores": scores,
        "qe": qe,
        "ge": ge,
        "recipe": recipe,
        "refusal": refusal,
        "retrieval_timing": {"total_retrieval_s": retrieval_s, "reranker_backend": "pytorch" if reranker_path else None},
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Production PT inference: base_full 0.65 + external_convnext_full 0.35"
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--deployment-dir", default="deploy/models_current")
    ap.add_argument("--out", default="outputs/test_submission")
    ap.add_argument("--device", default="0")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--warmup-batches", type=int, default=2)
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()

    inp = Path(a.input_dir).expanduser().resolve()
    dep = Path(a.deployment_dir).expanduser()
    if not dep.is_absolute():
        dep = (ROOT / dep).resolve()
    out = Path(a.out).expanduser()
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    meta = read_score_ensemble_deployment(dep)
    _validate_production(meta)
    qcsv, gcsv, images = inp / "test_query.csv", inp / "test_gallery.csv", inp / "images"
    if not qcsv.is_file() or not gcsv.is_file() or not images.is_dir():
        raise SystemExit("input-dir must contain test_query.csv, test_gallery.csv and images/")

    work = out / "work_score_ensemble"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    qprep, gprep = work / "query", work / "gallery"
    t0 = time.perf_counter()
    prepare_hackathon_dataset(qcsv, images, qprep, pad=.03, val_fraction=.0, eval_fraction=.0, materialize_crops=False)
    prepare_hackathon_dataset(gcsv, images, gprep, pad=.03, val_fraction=.0, eval_fraction=.0, materialize_crops=False)
    qmanifest = qprep / ("test.csv" if (qprep / "test.csv").is_file() else "manifest.csv")
    gmanifest = gprep / ("test.csv" if (gprep / "test.csv").is_file() else "manifest.csv")
    manifest_prepare_s = time.perf_counter() - t0

    qids = pd.read_csv(qcsv, dtype={"image_id": str})["image_id"].astype(str).tolist()
    gids = pd.read_csv(gcsv, dtype={"image_id": str})["image_id"].astype(str).tolist()

    specs = []
    for member in meta["members"]:
        mdir = Path(member["path"])
        dm = json.loads((mdir / "deployment.json").read_text(encoding="utf-8"))
        checkpoint = resolve(mdir, dm.get("checkpoint"), "reid.pt")
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError(f"{member['name']}: checkpoint missing: {checkpoint}")
        specs.append({"name": member["name"], "checkpoint": checkpoint})

    t_load = time.perf_counter()
    loaded_models, feature_device, image_size, load_info = load_multi_pt_models(specs, device=a.device)
    model_load_s = time.perf_counter() - t_load

    t_features = time.perf_counter()
    qpayloads, qstats = extract_feature_caches_loaded_pt_timed(
        qmanifest, loaded_models, feature_device, image_size, load_info["preprocess"],
        precision=a.precision, batch_size=a.batch, workers=a.workers, warmup_batches=a.warmup_batches,
    )
    gpayloads, gstats = extract_feature_caches_loaded_pt_timed(
        gmanifest, loaded_models, feature_device, image_size, load_info["preprocess"],
        precision=a.precision, batch_size=a.batch, workers=a.workers, warmup_batches=a.warmup_batches,
    )
    feature_total_s = time.perf_counter() - t_features

    matrices, weights, results = [], [], {}
    retrieval_total_s = 0.0
    reranker_device = _torch_device_arg(a.device)
    for member in meta["members"]:
        name = member["name"]
        qcache, gcache = qpayloads[name], gpayloads[name]
        if cache_ids(qcache) != qids or cache_ids(gcache) != gids:
            raise RuntimeError(f"{name}: cache row order does not match input CSV")
        res = _rank_member(name, member, qcache, gcache, reranker_device)
        retrieval_total_s += float(res["retrieval_timing"]["total_retrieval_s"])
        matrices.append(dense_final_scores(res["ranked"], res["scores"], qids, gids))
        weights.append(float(member["weight"]))
        results[name] = res

    fused = fuse_score_matrices(matrices, weights, meta.get("fusion", "per_query_zscore"))
    ranked = ranked_dict_from_matrix(fused, qids, gids)
    write_submission(out / "submission.csv", ranked, qids, top_k=10)

    policy = meta.get("candidate_policy", {}) or {}
    candidate_member = str(policy.get("member", "base_full"))
    if candidate_member not in results:
        raise RuntimeError(f"candidate member not evaluated: {candidate_member}")
    write_candidates(out / "candidates.csv", inherited_candidates(results[candidate_member], qids, gids))

    q_embedding = weighted_concat_embeddings([results[m["name"]]["qe"] for m in meta["members"]], weights)
    g_embedding = weighted_concat_embeddings([results[m["name"]]["ge"] for m in meta["members"]], weights)
    write_embeddings(out / "embeddings.npy", q_embedding, g_embedding)

    report = {
        "schema": "vehicle-reid-v094-production-pt-inference-v1",
        "deployment": str(dep),
        "runtime": "pytorch",
        "device": a.device,
        "precision": a.precision,
        "batch": a.batch,
        "workers": a.workers,
        "query_rows": len(qids),
        "gallery_rows": len(gids),
        "manifest_prepare_s": manifest_prepare_s,
        "model_load_s": model_load_s,
        "feature_total_s": feature_total_s,
        "feature_images_per_s": float((len(qids) + len(gids)) / max(feature_total_s, 1e-9)),
        "retrieval_total_s": retrieval_total_s,
        "fusion": meta.get("fusion"),
        "members": [{"name": m["name"], "weight": m["weight"]} for m in meta["members"]],
        "candidate_policy": policy,
        "embedding_policy": "weighted_concat(base_full=0.65,external_convnext_full=0.35)",
        "shared_preprocess": True,
        "query_extract": qstats,
        "gallery_extract": gstats,
        "member_retrieval": {k: v["retrieval_timing"] for k, v in results.items()},
        "notes": [
            "JPEG decode/BBox crop/resize/normalization/color descriptor are shared between both PT feature models.",
            "The two neural forwards remain independent and are executed sequentially on the selected GPU.",
            "submission.csv uses per-query z-score fusion of the two members' final ranking scores.",
            "candidates.csv inherits base_full refusal calibration because fused refusal was not recalibrated.",
            "embeddings.npy is the weighted concatenation of both members' normalized retrieval embeddings.",
        ],
    }
    (out / "score_ensemble_inference.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] {out / 'submission.csv'}")


if __name__ == "__main__":
    main()
