from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
from typing import Iterable

import pandas as pd


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VERI_FILENAME_RE = re.compile(r"(?P<vid>\d+)_c(?P<cam>\d+)", re.I)


def _sid(dataset: str, rel: str) -> str:
    return hashlib.sha1(f"{dataset}|{rel}".encode()).hexdigest()[:20]


def _iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*"), key=lambda x: x.as_posix()):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            yield p


def _veri_root_score(path: Path) -> int:
    score = 0
    if (path / "image_train").is_dir():
        score += 1000
    if (path / "bounding_box_train").is_dir():
        score += 900
    if (path / "image_query").is_dir():
        score += 150
    if (path / "query").is_dir():
        score += 120
    if (path / "image_test").is_dir():
        score += 150
    if (path / "bounding_box_test").is_dir():
        score += 120
    if (path / "gallery").is_dir():
        score += 100
    return score


def discover_veri776_root(root: str | Path) -> Path:
    """Resolve the actual VeRi-776 dataset root from common Kaggle/archive layouts.

    The Kaggle mirror may be unpacked directly into ``root`` or wrapped in one or more
    folders such as ``VeRi/``, ``VeRi_with_plate/`` or a dataset slug directory.  The
    only hard requirement is that an ``image_train`` or ``bounding_box_train`` folder
    exists somewhere below the supplied path.
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"VeRi source does not exist: {root}")

    # Allow users to point directly at image_train.
    if root.is_dir() and root.name.lower() in {"image_train", "bounding_box_train"}:
        root = root.parent

    candidates: dict[Path, int] = {}

    def add_candidate(p: Path) -> None:
        if not p.is_dir():
            return
        score = _veri_root_score(p)
        if score <= 0:
            return
        try:
            depth = len(p.relative_to(root).parts)
        except ValueError:
            depth = 99
        # Prefer a complete layout and, secondarily, a shallower wrapper.
        candidates[p] = score - depth

    add_candidate(root)

    # Search specifically for the train folder rather than recursively inspecting all
    # image files. This stays fast even when VeRi contains tens of thousands of images.
    for name in ("image_train", "bounding_box_train"):
        for d in root.rglob(name):
            if d.is_dir():
                add_candidate(d.parent)

    if not candidates:
        nearby = []
        for d in root.rglob("*"):
            if d.is_dir() and d.name.lower() in {
                "image_train", "bounding_box_train", "image_query", "query",
                "image_test", "bounding_box_test", "gallery"
            }:
                nearby.append(str(d))
                if len(nearby) >= 20:
                    break
        hint = "\n  ".join(nearby) if nearby else "<none found>"
        raise RuntimeError(
            "Could not locate a VeRi-776 training directory. Expected image_train/ "
            "or bounding_box_train/ somewhere below:\n"
            f"  {root}\nRecognized-looking directories:\n  {hint}"
        )

    return max(candidates, key=candidates.get)


def _choose_veri_split_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    return None


def import_veri776(root: str | Path, out_csv: str | Path) -> pd.DataFrame:
    """Import VeRi-776, including the common Kaggle mirror layout.

    Supported layouts include direct VeRi roots and arbitrary wrapper directories.
    The importer understands ``image_train``, ``image_query`` and ``image_test`` as
    well as the common aliases ``bounding_box_train``, ``query`` and
    ``bounding_box_test``.  Only the ``train`` split is used later for external ReID
    pretraining; query/gallery rows are retained in ``veri776.csv`` for auditing and
    optional official-protocol evaluation.
    """
    supplied_root = Path(root).expanduser().resolve()
    resolved_root = discover_veri776_root(supplied_root)

    split_specs = [
        ("train", ("image_train", "bounding_box_train")),
        ("query", ("image_query", "query")),
        ("gallery", ("image_test", "bounding_box_test", "gallery")),
    ]

    rows: list[dict] = []
    parse_failures: list[str] = []
    selected_dirs: dict[str, str | None] = {}

    for split, aliases in split_specs:
        d = _choose_veri_split_dir(resolved_root, aliases)
        selected_dirs[split] = str(d) if d is not None else None
        if d is None:
            continue
        for p in _iter_images(d):
            m = _VERI_FILENAME_RE.search(p.name)
            if not m:
                if len(parse_failures) < 50:
                    parse_failures.append(str(p))
                continue
            rel = p.relative_to(resolved_root).as_posix()
            vid = m.group("vid")
            cam = m.group("cam")
            rows.append({
                "sample_id": _sid("veri776", rel),
                "path": str(p.resolve()),
                "vehicle_id": vid,
                "vehicle_key": f"veri776:{vid}",
                "camera_id": cam,
                "dataset": "veri776",
                "split": split,
                "image_id": p.stem,
                "source_row": -1,
                "source_split_dir": d.name,
            })

    if not rows:
        raise RuntimeError(
            f"No VeRi-776 images with names like 0002_c002_... were recognized under {resolved_root}"
        )

    df = pd.DataFrame(rows).drop_duplicates(["path", "split"])
    split_order = pd.Categorical(df["split"], categories=["train", "query", "gallery"], ordered=True)
    df = (df.assign(_split_order=split_order)
            .sort_values(["_split_order", "vehicle_id", "camera_id", "path"], kind="stable")
            .drop(columns="_split_order")
            .reset_index(drop=True))
    train = df[df["split"] == "train"]
    if train.empty:
        raise RuntimeError(f"VeRi root was found at {resolved_root}, but no training images were imported")

    stats = {
        "supplied_root": str(supplied_root),
        "resolved_root": str(resolved_root),
        "selected_dirs": selected_dirs,
        "rows": int(len(df)),
        "rows_by_split": {str(k): int(v) for k, v in df.groupby("split").size().items()},
        "ids_by_split": {str(k): int(v) for k, v in df.groupby("split")["vehicle_id"].nunique().items()},
        "cameras_by_split": {str(k): int(v) for k, v in df.groupby("split")["camera_id"].nunique().items()},
        "train_rows": int(len(train)),
        "train_ids": int(train["vehicle_id"].nunique()),
        "train_cameras": int(train["camera_id"].nunique()),
        "parse_failure_examples": parse_failures,
    }
    df.attrs["veri776_import_report"] = stats

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def write_veri776_report(df: pd.DataFrame, path: str | Path) -> dict:
    report = dict(df.attrs.get("veri776_import_report", {}))
    if not report:
        report = {
            "rows": int(len(df)),
            "rows_by_split": {str(k): int(v) for k, v in df.groupby("split").size().items()},
            "ids_by_split": {str(k): int(v) for k, v in df.groupby("split")["vehicle_id"].nunique().items()},
            "cameras_by_split": {str(k): int(v) for k, v in df.groupby("split")["camera_id"].nunique().items()},
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


# Legacy importers are kept for backwards compatibility with earlier project archives.
# They are not part of the v0.3 default external-data workflow.
def _find_image(root: Path, rel: str) -> Path | None:
    p = root / rel
    if p.exists(): return p
    p = root / "images" / rel
    if p.exists(): return p
    hits = list(root.rglob(Path(rel).name))
    return hits[0] if hits else None


def _parse_vehicle_info(path: Path) -> dict[str, tuple[str, str]]:
    info = {}
    if not path.exists(): return info
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            toks = line.strip().replace(",", " ").split()
            if len(toks) < 2: continue
            key = toks[0].replace("\\", "/")
            nums = [t for t in toks[1:] if re.fullmatch(r"-?\d+", t)]
            vid = nums[0] if nums else Path(key).parent.name
            cam = nums[1] if len(nums) > 1 else "-1"
            info[key] = (vid, cam)
            info[Path(key).name] = (vid, cam)
    return info


def import_veriwild(root: str | Path, out_csv: str | Path) -> pd.DataFrame:
    root = Path(root)
    split_root = root / "train_test_split"
    if not split_root.exists() and (root / "data" / "train_test_split").exists():
        root = root / "data"; split_root = root / "train_test_split"
    info = _parse_vehicle_info(split_root / "vehicle_info.txt")
    lists = [("train_list.txt", "train")]
    rows = []
    for fname, split in lists:
        lp = split_root / fname
        if not lp.exists(): continue
        with open(lp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                toks = line.strip().split()
                if not toks: continue
                rel = toks[0].replace("\\", "/")
                p = _find_image(root, rel)
                if p is None: continue
                vid, cam = info.get(rel, info.get(Path(rel).name, (None, None)))
                nums = [t for t in toks[1:] if re.fullmatch(r"\d+", t)]
                if vid is None:
                    vid = nums[0] if nums else (Path(rel).parent.name or p.parent.name)
                if cam is None or cam == "-1":
                    cm = re.search(r"c(?:am)?[_-]?(\d+)", p.name, re.I)
                    cam = cm.group(1) if cm else (nums[1] if len(nums) > 1 else "-1")
                r = p.relative_to(root).as_posix() if p.is_relative_to(root) else p.name
                rows.append({"sample_id": _sid("veriwild", r), "path": str(p.resolve()),
                             "vehicle_id": str(vid), "camera_id": str(cam), "dataset": "veriwild",
                             "split": split, "image_id": p.stem, "source_row": -1})
    df = pd.DataFrame(rows).drop_duplicates("path")
    if df.empty:
        raise RuntimeError("Could not import VERI-Wild. Expected train_test_split/train_list.txt and extracted images.")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True); df.to_csv(out_csv, index=False)
    return df


def import_compcars(root: str | Path, out_csv: str | Path) -> pd.DataFrame:
    """Legacy optional importer; not used by the v0.3 default workflow."""
    root = Path(root)
    candidates = [root / "train_test_split" / "classification", root / "data" / "train_test_split" / "classification"]
    split_root = next((p for p in candidates if p.exists()), None)
    image_roots = [root / "image", root / "data" / "image"]
    image_root = next((p for p in image_roots if p.exists()), None)
    if split_root is None or image_root is None:
        if (root / "train_surveillance.txt").exists():
            split_root = root
            image_root = root / "image"
        else:
            raise RuntimeError("Unsupported CompCars layout; expected image/ and train_test_split/classification/.")
    rows = []
    for split in ("train", "test"):
        f = split_root / f"{split}.txt"
        if not f.exists(): f = split_root / f"{split}_surveillance.txt"
        if not f.exists(): continue
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            rel = line.strip().split()[0]
            p = image_root / rel
            if not p.exists(): continue
            parts = Path(rel).parts
            make_id = parts[0] if len(parts) > 0 else "-1"
            model_id = parts[1] if len(parts) > 1 else make_id
            class_id = f"{make_id}:{model_id}"
            rows.append({"sample_id": _sid("compcars", rel), "path": str(p.resolve()),
                         "vehicle_id": class_id, "camera_id": "-1", "image_id": p.stem, "source_row": -1,
                         "semantic_class": class_id, "make_id": make_id, "model_id": model_id,
                         "dataset": "compcars", "split": split})
    df = pd.DataFrame(rows)
    if df.empty: raise RuntimeError("No CompCars samples imported")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True); df.to_csv(out_csv, index=False)
    return df
