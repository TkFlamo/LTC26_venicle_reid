from __future__ import annotations

import csv
import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_evaluator_path() -> Path:
    return project_root() / "official" / "evaluate.py"


@lru_cache(maxsize=4)
def load_official_evaluator(path: str | Path | None = None) -> ModuleType:
    p = Path(path) if path is not None else default_evaluator_path()
    p = p.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Official evaluator not found: {p}")
    spec = importlib.util.spec_from_file_location("falcon_official_evaluate", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official evaluator: {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    return x


def blended_embedding(z_global: np.ndarray, z_fused: np.ndarray, alpha: float) -> np.ndarray:
    """Embedding whose dot product equals convex global/fused cosine fusion."""
    a = float(np.clip(alpha, 0.0, 1.0))
    zg = normalize(z_global); zf = normalize(z_fused)
    if a <= 0.0:
        return zg
    if a >= 1.0:
        return zf
    return np.concatenate([np.sqrt(1.0-a)*zg, np.sqrt(a)*zf], axis=1).astype(np.float32)


def ranked_from_similarity(q_ids: list[str], g_ids: list[str], sim: np.ndarray, limit: int | None = None) -> dict[str, list[str]]:
    q_ids = [str(x) for x in q_ids]; g_ids = np.asarray([str(x) for x in g_ids], dtype=str)
    out: dict[str, list[str]] = {}
    for i, qid in enumerate(q_ids):
        order = np.argsort(-sim[i], kind="stable")
        if limit is not None:
            order = order[:int(limit)]
        out[qid] = g_ids[order].tolist()
    return out


def protocol_ids(query_csv: str | Path, gallery_csv: str | Path) -> tuple[list[str], list[str]]:
    q = pd.read_csv(query_csv, dtype={"image_id": str})
    g = pd.read_csv(gallery_csv, dtype={"image_id": str})
    if "image_id" not in q.columns or "image_id" not in g.columns:
        raise ValueError("Official query/gallery manifests must contain image_id")
    return q.image_id.astype(str).tolist(), g.image_id.astype(str).tolist()


def official_metrics_from_similarity(
    sim: np.ndarray,
    *,
    q_ids: list[str],
    g_ids: list[str],
    gt_csv: str | Path,
    q_emb: np.ndarray | None = None,
    g_emb: np.ndarray | None = None,
    top_k: int = 10,
    evaluator_path: str | Path | None = None,
) -> dict:
    """Run the organizers' own ranking functions without changing their formulas."""
    mod = load_official_evaluator(evaluator_path)
    query, gallery = mod.load_gt(str(gt_csv))
    # submission.csv formally carries top-K predictions only, so model selection simulates exactly
    # what the unchanged organizer script will read from the generated file.
    ranked = ranked_from_similarity(q_ids, g_ids, np.asarray(sim, dtype=np.float32), limit=int(top_k))
    ranking = mod.ranking_metrics(query, gallery, ranked, top_k=int(top_k), ranks=(1,5))
    out = {"ranking": ranking}
    if q_emb is not None and g_emb is not None:
        full = mod.full_ranking_metrics(
            normalize(q_emb), normalize(g_emb), [str(x) for x in q_ids], [str(x) for x in g_ids], query, gallery
        )
        out["full_ranking"] = full
    return out


def official_metrics_from_embeddings(
    q_emb: np.ndarray,
    g_emb: np.ndarray,
    *,
    q_ids: list[str],
    g_ids: list[str],
    gt_csv: str | Path,
    top_k: int = 10,
    evaluator_path: str | Path | None = None,
) -> dict:
    q = normalize(q_emb); g = normalize(g_emb)
    return official_metrics_from_similarity(
        q @ g.T, q_ids=q_ids, g_ids=g_ids, gt_csv=gt_csv,
        q_emb=q, g_emb=g, top_k=top_k, evaluator_path=evaluator_path,
    )


def flatten_official_report(report: dict, prefix: str = "official_val") -> dict[str, float | int]:
    r = report.get("ranking", {})
    f = report.get("full_ranking", {})
    return {
        f"{prefix}_mAP@10": float(r.get("mAP@10", 0.0)),
        f"{prefix}_Rank-1": float(r.get("Rank-1", 0.0)),
        f"{prefix}_Rank-5": float(r.get("Rank-5", 0.0)),
        f"{prefix}_ranking_queries": int(r.get("n_scored", 0)),
        f"{prefix}_openset_excluded": int(r.get("n_openset_excluded", 0)),
        f"{prefix}_mAP_full": float(f.get("mAP_full", 0.0)),
        f"{prefix}_mINP": float(f.get("mINP", 0.0)),
    }


def official_selection_key(report: dict) -> tuple[float, float, float, float, float]:
    """Primary metric first, then only official diagnostics as deterministic tie-breakers."""
    r = report.get("ranking", {}); f = report.get("full_ranking", {})
    return (
        float(r.get("mAP@10", 0.0)),
        float(r.get("Rank-1", 0.0)),
        float(r.get("Rank-5", 0.0)),
        float(f.get("mAP_full", 0.0)),
        float(f.get("mINP", 0.0)),
    )


def write_submission(path: str | Path, ranked: dict[str, list[str]], query_order: list[str], top_k: int = 10) -> Path:
    """Exact organizer format: no header, query_id,gallery_id_1,..."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for qid in query_order:
            w.writerow([str(qid), *[str(x) for x in ranked.get(str(qid), [])[:int(top_k)]]])
    return path


def write_candidates(path: str | Path, rows: list[tuple[str, str, float]]) -> Path:
    """Exact organizer format. Refusal is represented by no row for that query."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["query_id", "gallery_id", "confidence"]).to_csv(path, index=False)
    return path


def write_embeddings(path: str | Path, q_emb: np.ndarray, g_emb: np.ndarray) -> Path:
    """Official row order: all query rows first, then all gallery rows."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.concatenate([normalize(q_emb), normalize(g_emb)], axis=0).astype(np.float32))
    return path
