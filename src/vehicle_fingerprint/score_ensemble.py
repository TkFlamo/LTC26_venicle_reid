from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def dense_final_scores(ranked: dict, scores: dict, qids: list[str], gids: list[str]) -> np.ndarray:
    """Convert per-query ranked lists + scores into a dense [Q,G] score matrix.

    This intentionally matches scripts/55_eval_no_train_exhaustive_shared_v094.py,
    including the deterministic fill value for any gallery ids omitted by a member.
    """
    gidx = {str(g): i for i, g in enumerate(gids)}
    out = np.full((len(qids), len(gids)), np.nan, np.float32)
    for qi, qid in enumerate(qids):
        rr = ranked.get(str(qid), [])
        ss = np.asarray(scores.get(str(qid), []), dtype=np.float32).reshape(-1)
        n = min(len(rr), len(ss))
        for gid, score in zip(rr[:n], ss[:n]):
            j = gidx.get(str(gid))
            if j is not None:
                out[qi, j] = float(score)
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


def fuse_score_matrices(matrices: list[np.ndarray], weights: list[float], fusion: str = "per_query_zscore") -> np.ndarray:
    if not matrices:
        raise ValueError("No score matrices supplied")
    if len(matrices) != len(weights):
        raise ValueError("matrices/weights length mismatch")
    shape = matrices[0].shape
    if any(np.asarray(x).shape != shape for x in matrices):
        raise ValueError("All score matrices must have the same shape")
    w = np.asarray(weights, np.float32)
    if np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("weights must be non-negative and have positive sum")
    w = w / w.sum()
    f = str(fusion).lower()
    if f != "per_query_zscore":
        raise ValueError(f"Unsupported score fusion: {fusion}")
    out = np.zeros(shape, np.float32)
    for wi, matrix in zip(w.tolist(), matrices):
        out += float(wi) * zscore_rows(matrix)
    return out


def ranked_dict_from_matrix(x: np.ndarray, qids: list[str], gids: list[str]) -> dict[str, list[str]]:
    order = np.argsort(-np.asarray(x), axis=1, kind="stable")
    return {str(qid): [str(gids[j]) for j in order[i]] for i, qid in enumerate(qids)}


def read_score_ensemble_deployment(root: str | Path) -> dict:
    root = Path(root).expanduser().resolve()
    meta_path = root / "deployment.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("mode", "")).lower() != "score_ensemble":
        raise ValueError(f"Expected deployment mode=score_ensemble, got {meta.get('mode')!r}")
    members = meta.get("members") or []
    if len(members) < 2:
        raise ValueError("score_ensemble deployment needs at least two members")
    total = 0.0
    resolved = []
    names = set()
    for item in members:
        name = str(item["name"])
        if name in names:
            raise ValueError(f"Duplicate member name: {name}")
        names.add(name)
        weight = float(item["weight"])
        p = Path(item["path"]).expanduser()
        if not p.is_absolute():
            p = root / p
        p = p.resolve()
        if not (p / "deployment.json").is_file():
            raise FileNotFoundError(f"{name}: missing deployment.json under {p}")
        resolved.append({**item, "name": name, "weight": weight, "path": p})
        total += weight
    if total <= 0:
        raise ValueError("Ensemble weight sum must be positive")
    for item in resolved:
        item["weight"] = float(item["weight"] / total)
    return {**meta, "root": root, "members": resolved}


def weighted_concat_embeddings(embeddings: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Create one honest fixed-dimensional embedding from several deployed members.

    Each member embedding is L2-normalized, scaled by sqrt(weight), and concatenated.
    The resulting cosine similarity equals the weighted sum of member cosine similarities.
    Final submission ranking may still apply documented member-specific reranking and
    per-query score z-normalization on top of these representations.
    """
    if not embeddings:
        raise ValueError("No embeddings supplied")
    if len(embeddings) != len(weights):
        raise ValueError("embeddings/weights length mismatch")
    n = int(np.asarray(embeddings[0]).shape[0])
    if any(np.asarray(x).ndim != 2 or int(np.asarray(x).shape[0]) != n for x in embeddings):
        raise ValueError("All embeddings must be 2D with the same row count")
    w = np.asarray(weights, np.float32)
    if np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("weights must be non-negative and have positive sum")
    w = w / w.sum()
    parts = []
    for wi, x in zip(w.tolist(), embeddings):
        a = np.asarray(x, np.float32)
        a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)
        parts.append(np.sqrt(float(wi)) * a)
    out = np.concatenate(parts, axis=1).astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True).clip(1e-12)
    return out
