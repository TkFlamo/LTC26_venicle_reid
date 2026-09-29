from __future__ import annotations

"""Leakage-resistant development K-fold construction for vehicle ReID.

The competition metric is unusually sensitive to the exact query/gallery composition.  A single
validation split is therefore too noisy for architecture search.  This module keeps an untouched
identity holdout and partitions the remaining identities into source-image-disjoint folds while
balancing the quantities that materially change ReID difficulty: number of images, number of
cameras, cross-camera eligibility, bbox scale/aspect and image brightness.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageStat

from .official_protocol import build_official_protocol
from .splits import _identity_components, assert_split_disjoint


@dataclass
class FoldBuildConfig:
    n_folds: int = 4
    holdout_fraction: float = 0.10
    seed: int = 42
    min_eval_cameras: int = 2
    open_set_fraction: float = 0.20
    max_queries_per_id: int = 2
    compute_brightness: bool = True


def _safe_float(v, default=0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _mean_brightness(path: str | Path) -> float:
    """Cheap deterministic luma estimate on a tiny thumbnail."""
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            im.thumbnail((48, 48), Image.Resampling.BILINEAR)
            return float(ImageStat.Stat(im).mean[0] / 255.0)
    except Exception:
        return float("nan")


def enrich_manifest(df: pd.DataFrame, *, compute_brightness: bool = True) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    for c in ("vehicle_id", "camera_id", "image_id"):
        if c in out:
            out[c] = out[c].astype(str)
    if all(c in out for c in ("w", "h")):
        w = pd.to_numeric(out["w"], errors="coerce").clip(lower=1e-6)
        h = pd.to_numeric(out["h"], errors="coerce").clip(lower=1e-6)
        out["bbox_area"] = (w * h).astype(float)
        out["bbox_aspect"] = (w / h).astype(float)
    else:
        out["bbox_area"] = 1.0
        out["bbox_aspect"] = 1.0
    if "brightness" not in out.columns:
        if compute_brightness:
            vals = []
            for row in out.itertuples(index=False):
                p = getattr(row, "path", None) or getattr(row, "source_path", None)
                vals.append(_mean_brightness(p) if p else float("nan"))
            med = float(np.nanmedian(vals)) if np.isfinite(np.asarray(vals, float)).any() else 0.5
            out["brightness"] = np.where(np.isfinite(vals), vals, med)
        else:
            out["brightness"] = 0.5
    return out


def _component_records(df: pd.DataFrame, min_eval_cameras: int) -> list[dict]:
    comps = _identity_components(df)
    id_stats = df.groupby("vehicle_id").agg(
        n_images=("vehicle_id", "size"),
        n_cameras=("camera_id", "nunique"),
        bbox_area=("bbox_area", "median"),
        bbox_aspect=("bbox_aspect", "median"),
        brightness=("brightness", "median"),
    )
    recs = []
    for root, members in comps.items():
        g = df[df.vehicle_id.isin(members)]
        s = id_stats.loc[members]
        recs.append({
            "root": str(root),
            "members": list(map(str, members)),
            "n_ids": int(len(members)),
            "rows": int(len(g)),
            "source_images": int(g.image_id.astype(str).nunique()),
            "cross_ids": int((s.n_cameras >= int(min_eval_cameras)).sum()),
            "mean_cameras": float(s.n_cameras.mean()),
            "log_bbox_area": float(np.log(np.clip(s.bbox_area.astype(float), 1e-6, None)).mean()),
            "log_aspect": float(np.log(np.clip(s.bbox_aspect.astype(float), 1e-6, None)).mean()),
            "brightness": float(s.brightness.astype(float).mean()),
        })
    return recs


def _normalise_feature(recs: list[dict], key: str) -> dict[str, float]:
    vals = np.asarray([r[key] for r in recs], dtype=float)
    mu = float(np.mean(vals)); sd = float(np.std(vals))
    if sd < 1e-8: sd = 1.0
    return {r["root"]: float((r[key] - mu) / sd) for r in recs}


def _choose_holdout(recs: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Choose a representative untouched holdout rather than simply the hardest components.

    Components are indivisible because identities sharing one source frame must stay together.  We
    greedily match the requested fraction of the *global* dataset totals for identity count, rows,
    cross-camera eligibility and coarse appearance/capture statistics.  This makes the final holdout
    less likely to be accidentally easier/harder than the development folds.
    """
    if frac <= 0:
        return [], list(recs)
    if not recs:
        return [], []
    rng = np.random.default_rng(seed)
    # Additive proxy totals. Continuous statistics are identity-weighted so matching their sums also
    # matches their approximate means once identity cardinality is close to target.
    def vec(r):
        w=max(1.0,float(r["n_ids"]))
        return np.asarray([
            float(r["n_ids"]), float(r["rows"]), float(r["cross_ids"]),
            w*float(r["mean_cameras"]), w*float(r["log_bbox_area"]),
            w*float(r["log_aspect"]), w*float(r["brightness"]),
        ],dtype=float)
    V=np.stack([vec(r) for r in recs]); total=V.sum(0); target=np.maximum(np.abs(total*float(frac)),1e-6)
    target_ids=max(1,int(round(total[0]*float(frac))))
    # Cardinalities dominate; visual/capture distribution terms act as secondary stratifiers.
    weights=np.asarray([2.0,1.25,1.5,0.30,0.20,0.20,0.25],dtype=float)
    jitter={r["root"]:float(rng.random()) for r in recs}
    selected=[]; current=np.zeros_like(total); remaining=list(range(len(recs)))
    while remaining and (current[0] < target_ids or not selected):
        best=None
        for idx in remaining:
            nv=current+V[idx]
            rel=(nv-total*float(frac))/target
            # Strongly discourage huge identity/row overshoot but permit indivisible components.
            overs=np.maximum(rel[:3],0.0)
            score=float((weights*rel*rel).sum()+0.35*(overs*overs).sum())
            cand=(score,jitter[recs[idx]["root"]],idx)
            if best is None or cand<best: best=cand
        idx=int(best[2]);selected.append(recs[idx]);current+=V[idx];remaining.remove(idx)
    roots={r["root"] for r in selected}
    return selected,[r for r in recs if r["root"] not in roots]

def _assign_folds(recs: list[dict], n_folds: int, seed: int) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("n_folds must be >=2")
    rng = np.random.default_rng(seed)
    features = {
        k: _normalise_feature(recs, k)
        for k in ("mean_cameras", "log_bbox_area", "log_aspect", "brightness")
    }
    jitter = {r["root"]: float(rng.random()) for r in recs}
    # Hard/large components first, random only for ties.
    order = sorted(
        recs,
        key=lambda r: (r["cross_ids"], r["n_ids"], r["rows"], jitter[r["root"]]),
        reverse=True,
    )
    target_ids = sum(r["n_ids"] for r in recs) / n_folds
    target_rows = sum(r["rows"] for r in recs) / n_folds
    target_cross = max(1e-6, sum(r["cross_ids"] for r in recs) / n_folds)
    state = [dict(ids=0.0, rows=0.0, cross=0.0, feat=defaultdict(float), ncomp=0) for _ in range(n_folds)]
    out: dict[str, int] = {}
    for r in order:
        best = None
        for f in range(n_folds):
            st = state[f]
            ids = (st["ids"] + r["n_ids"]) / max(target_ids, 1e-6)
            rows = (st["rows"] + r["rows"]) / max(target_rows, 1e-6)
            cross = (st["cross"] + r["cross_ids"]) / target_cross
            # Keep feature means similar without overpowering cardinality balance.
            nnew = st["ncomp"] + 1
            feat_pen = 0.0
            for k, zmap in features.items():
                mean_after = (st["feat"][k] + zmap[r["root"]]) / nnew
                feat_pen += mean_after * mean_after
            score = 1.20 * ids * ids + rows * rows + 1.35 * cross * cross + 0.12 * feat_pen
            # Tiny fold index tie break keeps deterministic behaviour.
            cand = (score, st["ncomp"], f)
            if best is None or cand < best:
                best = cand
        f = int(best[2]); out[r["root"]] = f
        st = state[f]; st["ids"] += r["n_ids"]; st["rows"] += r["rows"]; st["cross"] += r["cross_ids"]; st["ncomp"] += 1
        for k, zmap in features.items(): st["feat"][k] += zmap[r["root"]]
    return out


def build_kfold_manifests(
    manifest: str | Path | pd.DataFrame,
    out_dir: str | Path,
    *,
    n_folds: int = 4,
    holdout_fraction: float = 0.10,
    seed: int = 42,
    min_eval_cameras: int = 2,
    open_set_fraction: float = 0.20,
    max_queries_per_id: int = 2,
    compute_brightness: bool = True,
) -> dict:
    df = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
    required = {"vehicle_id", "camera_id", "image_id"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"K-fold build requires {sorted(miss)}")
    df = enrich_manifest(df, compute_brightness=compute_brightness)
    recs = _component_records(df, min_eval_cameras)
    hold, dev = _choose_holdout(recs, holdout_fraction, seed)
    fold_by_root = _assign_folds(dev, int(n_folds), seed + 17)

    root_for_id = {}
    for r in recs:
        for vid in r["members"]:
            root_for_id[str(vid)] = r["root"]
    hold_roots = {r["root"] for r in hold}
    df["cv_role"] = ["holdout" if root_for_id[str(v)] in hold_roots else "dev" for v in df.vehicle_id.astype(str)]
    df["cv_fold"] = [(-1 if root_for_id[str(v)] in hold_roots else int(fold_by_root[root_for_id[str(v)]])) for v in df.vehicle_id.astype(str)]

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "manifest_enriched.csv", index=False)
    hold_df = df[df.cv_role == "holdout"].reset_index(drop=True)
    dev_df = df[df.cv_role == "dev"].reset_index(drop=True)
    hold_df.to_csv(out / "holdout.csv", index=False); dev_df.to_csv(out / "dev.csv", index=False)

    fold_reports = []
    for f in range(int(n_folds)):
        fd = out / f"fold_{f}"; fd.mkdir(parents=True, exist_ok=True)
        va = dev_df[dev_df.cv_fold == f].copy().reset_index(drop=True)
        tr = dev_df[dev_df.cv_fold != f].copy().reset_index(drop=True)
        tr["split"] = "train"; va["split"] = "val"
        # assert helper expects a single frame with split labels.
        assert_split_disjoint(pd.concat([tr, va], ignore_index=True))
        tr.to_csv(fd / "train.csv", index=False); va.to_csv(fd / "val.csv", index=False)
        proto = build_official_protocol(
            va, fd / "official_val", seed=seed + 1000 + f,
            open_set_fraction=open_set_fraction, max_queries_per_id=max_queries_per_id,
        )
        fold_reports.append({
            "fold": f, "train_rows": int(len(tr)), "val_rows": int(len(va)),
            "train_ids": int(tr.vehicle_id.nunique()), "val_ids": int(va.vehicle_id.nunique()),
            **{f"protocol_{k}": v for k, v in proto.items() if isinstance(v, (int, float))},
        })

    if len(hold_df):
        build_official_protocol(
            hold_df, out / "official_holdout", seed=seed + 9000,
            open_set_fraction=open_set_fraction, max_queries_per_id=max_queries_per_id,
        )

    summary = {
        "n_folds": int(n_folds), "holdout_fraction": float(holdout_fraction), "seed": int(seed),
        "rows": int(len(df)), "identities": int(df.vehicle_id.nunique()),
        "dev_rows": int(len(dev_df)), "dev_ids": int(dev_df.vehicle_id.nunique()),
        "holdout_rows": int(len(hold_df)), "holdout_ids": int(hold_df.vehicle_id.nunique()),
        "folds": fold_reports,
        "stratification": ["n_images", "n_cameras", "cross_camera_eligibility", "bbox_area", "bbox_aspect", "brightness"],
    }
    (out / "kfold_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary



def _sha_lines(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def load_fixed_validation_spec(path: str | Path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if "ordered_validation_rows" not in spec or "official_protocol" not in spec:
        raise ValueError(f"Invalid fixed validation spec: {path}")
    return spec


def build_v5_shared_protocols(shared_val: pd.DataFrame, out_dir: str | Path, spec: dict) -> dict:
    """Materialize the exact 4-fold local validation protocol used by vehicle_reid_v5_official.

    This intentionally mirrors ``make_official_val_folds`` from the comparison project.  The
    *validation pool* and its row order are fixed by ``ordered_validation_rows`` in the spec,
    while the number of *inner training folds* in this project remains independent.
    """
    df = shared_val.reset_index(drop=True).copy()
    for c in ("image_id", "vehicle_id", "camera_id"):
        df[c] = df[c].astype(str)
    cfg = spec["official_protocol"]
    folds = int(cfg.get("folds", 4))
    seed = int(cfg.get("seed", 59))
    open_set_ratio = float(cfg.get("open_set_ratio", .20))
    expected = {int(x["fold"]): x for x in cfg.get("fold_fingerprints", [])}

    groups = {str(pid): list(g.index) for pid, g in df.groupby(df.vehicle_id.astype(str), sort=True)}
    eligible = []
    for pid, inds in groups.items():
        if df.iloc[inds].camera_id.astype(str).nunique() >= 2:
            eligible.append(pid)
    if len(eligible) < 2:
        raise ValueError("Shared validation requires at least two multi-camera identities")

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    all_idx = np.arange(len(df), dtype=np.int64)
    reports = []
    for f in range(folds):
        rng = random.Random(seed + f * 7919)
        pids = eligible.copy(); rng.shuffle(pids)
        n_open = int(round(len(pids) * open_set_ratio))
        n_open = min(max(1, n_open), max(1, len(pids) - 1)) if open_set_ratio > 0 else 0
        open_pids = set(pids[:n_open])
        qidx = []
        for pid in eligible:
            inds = groups[pid]
            by_cam = defaultdict(list)
            for i in inds:
                by_cam[str(df.iloc[i].camera_id)].append(i)
            cams = sorted(by_cam)
            cam = cams[f % len(cams)]
            choices = by_cam[cam]
            qidx.append(choices[(f // len(cams)) % len(choices)])
        qset = set(qidx)
        gidx = [int(i) for i in all_idx if int(i) not in qset and str(df.iloc[int(i)].vehicle_id) not in open_pids]

        qdf = df.iloc[qidx].copy().reset_index(drop=True)
        gdf = df.iloc[gidx].copy().reset_index(drop=True)
        fd = out / f"fold_{f}"; fd.mkdir(parents=True, exist_ok=True)
        qdf.to_csv(fd / "query.csv", index=False)
        gdf.to_csv(fd / "gallery.csv", index=False)
        qgt = qdf[["image_id", "vehicle_id", "camera_id"]].copy(); qgt["split"] = "query"
        ggt = gdf[["image_id", "vehicle_id", "camera_id"]].copy(); ggt["split"] = "gallery"
        pd.concat([qgt, ggt], ignore_index=True).to_csv(fd / "ground_truth.csv", index=False)

        qhash = _sha_lines(qdf.image_id.astype(str).tolist())
        ghash = _sha_lines(gdf.image_id.astype(str).tolist())
        exp = expected.get(f)
        if exp:
            if qhash != exp.get("query_image_ids_sha256") or ghash != exp.get("gallery_image_ids_sha256"):
                raise AssertionError(
                    f"Shared validation protocol fold {f} does not match vehicle_reid_v5_official fingerprints"
                )
        reports.append({
            "fold": f, "query_rows": int(len(qdf)), "gallery_rows": int(len(gdf)),
            "open_set_ids": int(len(open_pids)), "query_image_ids_sha256": qhash,
            "gallery_image_ids_sha256": ghash,
        })
    summary = {"folds": folds, "seed": seed, "open_set_ratio": open_set_ratio, "reports": reports}
    (out / "protocol_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_shared_validation_kfold_manifests(
    manifest: str | Path | pd.DataFrame,
    out_dir: str | Path,
    *,
    validation_spec: str | Path,
    n_folds: int = 4,
    seed: int = 42,
    min_eval_cameras: int = 2,
    open_set_fraction: float = .20,
    max_queries_per_id: int = 2,
    compute_brightness: bool = True,
) -> dict:
    """Use a fixed cross-project validation pool, then K-fold only the remaining development IDs.

    The shared validation set is reconstructed from exact ordered image IDs exported from
    ``vehicle_reid_v5_official``.  Thus two developers may use different inner K values while the
    final benchmark stays byte-for-byte comparable at the query/gallery ID level.
    """
    raw = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
    required = {"vehicle_id", "camera_id", "image_id"}
    miss = required - set(raw.columns)
    if miss:
        raise ValueError(f"K-fold build requires {sorted(miss)}")
    for c in ("vehicle_id", "camera_id", "image_id"):
        raw[c] = raw[c].astype(str)
    if raw.image_id.duplicated().any():
        raise ValueError("Fixed shared validation expects unique image_id rows")

    spec = load_fixed_validation_spec(validation_spec)
    ordered = spec["ordered_validation_rows"]
    val_ids_ordered = [str(r["image_id"]) for r in ordered]
    row_by_image = raw.set_index("image_id", drop=False)
    missing_images = [x for x in val_ids_ordered if x not in row_by_image.index]
    if missing_images:
        raise ValueError(f"Shared validation mismatch: {len(missing_images)} expected image_id values are missing; first={missing_images[:5]}")

    shared_val = row_by_image.loc[val_ids_ordered].copy().reset_index(drop=True)
    # Verify labels/cameras too, so a different train.csv revision cannot silently masquerade as the same benchmark.
    for i, exp in enumerate(ordered):
        got = shared_val.iloc[i]
        if str(got.vehicle_id) != str(exp["vehicle_id"]) or str(got.camera_id) != str(exp["camera_id"]):
            raise AssertionError(
                f"Shared validation metadata mismatch for image_id={exp['image_id']}: "
                f"expected vehicle/camera={exp['vehicle_id']}/{exp['camera_id']} got={got.vehicle_id}/{got.camera_id}"
            )
    shared_images = set(val_ids_ordered)
    dev = raw[~raw.image_id.isin(shared_images)].copy().reset_index(drop=True)

    exp_counts = spec.get("dataset_expectations", {})
    checks = {
        "total_rows": len(raw), "total_vehicle_ids": raw.vehicle_id.nunique(),
        "development_rows": len(dev), "development_vehicle_ids": dev.vehicle_id.nunique(),
        "shared_validation_rows": len(shared_val), "shared_validation_vehicle_ids": shared_val.vehicle_id.nunique(),
    }
    for k, got in checks.items():
        if k in exp_counts and int(got) != int(exp_counts[k]):
            raise AssertionError(f"Shared validation dataset mismatch: {k} expected={exp_counts[k]} got={got}")

    val_vids = set(shared_val.vehicle_id.astype(str))
    dev_vids = set(dev.vehicle_id.astype(str))
    if val_vids & dev_vids:
        raise AssertionError(f"Identity leakage into shared validation: {len(val_vids & dev_vids)} vehicle IDs")
    if set(shared_val.image_id) & set(dev.image_id):
        raise AssertionError("Source-image leakage into shared validation")

    fps = spec.get("fingerprints", {})
    actual_fp = {
        "val_vehicle_ids_sha256": _sha_lines(sorted(val_vids)),
        "val_image_ids_sorted_sha256": _sha_lines(sorted(shared_val.image_id.astype(str).tolist())),
        "val_image_ids_ordered_sha256": _sha_lines(shared_val.image_id.astype(str).tolist()),
        "dev_vehicle_ids_sha256": _sha_lines(sorted(dev_vids)),
        "dev_image_ids_sorted_sha256": _sha_lines(sorted(dev.image_id.astype(str).tolist())),
    }
    for k, got in actual_fp.items():
        if k in fps and got != fps[k]:
            raise AssertionError(f"Shared validation fingerprint mismatch for {k}")

    # Enrichment and inner folds happen only inside development data.  Shared benchmark rows are never touched.
    dev = enrich_manifest(dev, compute_brightness=compute_brightness)
    shared_val = enrich_manifest(shared_val, compute_brightness=compute_brightness)
    recs = _component_records(dev, min_eval_cameras)
    fold_by_root = _assign_folds(recs, int(n_folds), seed + 17)
    root_for_id = {}
    for r in recs:
        for vid in r["members"]:
            root_for_id[str(vid)] = r["root"]
    dev["cv_role"] = "dev"
    dev["cv_fold"] = [int(fold_by_root[root_for_id[str(v)]]) for v in dev.vehicle_id.astype(str)]
    shared_val["cv_role"] = "shared_validation"
    shared_val["cv_fold"] = -1

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.concat([dev, shared_val], ignore_index=True).to_csv(out / "manifest_enriched.csv", index=False)
    dev.to_csv(out / "dev.csv", index=False)
    shared_val.to_csv(out / "shared_val.csv", index=False)

    fold_reports = []
    for f in range(int(n_folds)):
        fd = out / f"fold_{f}"; fd.mkdir(parents=True, exist_ok=True)
        va = dev[dev.cv_fold == f].copy().reset_index(drop=True)
        tr = dev[dev.cv_fold != f].copy().reset_index(drop=True)
        tr["split"] = "train"; va["split"] = "val"
        assert_split_disjoint(pd.concat([tr, va], ignore_index=True))
        tr.to_csv(fd / "train.csv", index=False); va.to_csv(fd / "val.csv", index=False)
        # This protocol is an INTERNAL early-stopping/model-selection fold.  It is not the shared benchmark.
        proto = build_official_protocol(
            va, fd / "official_val", seed=seed + 1000 + f,
            open_set_fraction=open_set_fraction, max_queries_per_id=max_queries_per_id,
        )
        fold_reports.append({
            "fold": f, "train_rows": int(len(tr)), "val_rows": int(len(va)),
            "train_ids": int(tr.vehicle_id.nunique()), "val_ids": int(va.vehicle_id.nunique()),
            **{f"protocol_{k}": v for k, v in proto.items() if isinstance(v, (int, float))},
        })

    shared_protocol = build_v5_shared_protocols(shared_val, out / "shared_validation", spec)
    summary = {
        "mode": "fixed_shared_validation_plus_inner_kfold",
        "validation_spec": str(validation_spec), "n_folds": int(n_folds), "seed": int(seed),
        "rows": int(len(raw)), "identities": int(raw.vehicle_id.nunique()),
        "dev_rows": int(len(dev)), "dev_ids": int(dev.vehicle_id.nunique()),
        "shared_validation_rows": int(len(shared_val)), "shared_validation_ids": int(shared_val.vehicle_id.nunique()),
        "inner_folds": fold_reports, "shared_validation_protocol": shared_protocol,
        "shared_validation_fingerprints": actual_fp,
        "fair_comparison_note": "Inner K may differ across developers. Cross-project metrics must be reported only after refitting on all dev.csv and evaluating data/processed/hackathon_cv/shared_validation/fold_0..N.",
    }
    (out / "kfold_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
