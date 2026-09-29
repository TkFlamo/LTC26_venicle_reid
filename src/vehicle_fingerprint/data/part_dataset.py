from __future__ import annotations

from pathlib import Path
import os
import shutil

import yaml

from .schema import PART_CLASSES, PART_CLASS_TO_ID

# Ultralytics Carparts-Seg source classes -> our stable coarse ontology.
CARPARTS_MAP = {
    0: "rear_bumper",      # back_bumper
    1: "door",             # back_door
    2: "rear_glass",
    3: "door",
    4: "rear_light",
    5: "rear_light",
    6: "door",
    7: "rear_light",
    8: "front_bumper",
    9: "door",
    10: "front_glass",
    11: "door",
    12: "front_light",
    13: "front_light",
    14: "door",
    15: "front_light",
    16: "hood",
    17: "mirror",
    18: None,               # generic object: too ambiguous
    19: "mirror",
    20: "trunk_tailgate",
    21: "trunk_tailgate",
    22: "wheel",
}


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def _find_layout(root: Path, split: str):
    candidates = [
        (root / "images" / split, root / "labels" / split),
        (root / split / "images", root / split / "labels"),
        (root / "images" / ("valid" if split == "val" else split), root / "labels" / ("valid" if split == "val" else split)),
        (root / ("valid" if split == "val" else split) / "images", root / ("valid" if split == "val" else split) / "labels"),
    ]
    return next(((i,l) for i,l in candidates if i.exists() and l.exists()), (None,None))


def convert_carparts(carparts_root: str | Path, out_root: str | Path) -> dict[str, int]:
    root, out = Path(carparts_root), Path(out_root)
    stats = {}
    for split in ("train", "val", "test"):
        img_dir, lab_dir = _find_layout(root, split)
        if img_dir is None: continue
        n = 0
        out_split = "val" if split in {"val", "test"} else "train"
        for img in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")):
            lab = lab_dir / f"{img.stem}.txt"
            if not lab.exists(): continue
            prefix = f"carparts_{split}_{img.stem}"
            dst_img = out / "images" / out_split / f"{prefix}{img.suffix.lower()}"
            dst_lab = out / "labels" / out_split / f"{prefix}.txt"
            new_lines = []
            for line in lab.read_text(encoding="utf-8").splitlines():
                toks = line.split()
                if len(toks) < 7: continue
                src_cls = int(float(toks[0])); target_name = CARPARTS_MAP.get(src_cls)
                if target_name is None: continue
                new_lines.append(" ".join([str(PART_CLASS_TO_ID[target_name]), *toks[1:]]))
            if not new_lines: continue
            _link_or_copy(img, dst_img)
            dst_lab.parent.mkdir(parents=True, exist_ok=True)
            dst_lab.write_text("\n".join(new_lines)+"\n", encoding="utf-8")
            n += 1
        stats[split] = n
    write_yolo_yaml(out)
    return stats


def write_yolo_yaml(root: str | Path) -> Path:
    root = Path(root)
    p = root / "dataset.yaml"
    spec = {"path": str(root.resolve()), "train": "images/train", "val": "images/val", "names": {i:n for i,n in enumerate(PART_CLASSES)}}
    with open(p,"w",encoding="utf-8") as f: yaml.safe_dump(spec,f,sort_keys=False,allow_unicode=True)
    return p
