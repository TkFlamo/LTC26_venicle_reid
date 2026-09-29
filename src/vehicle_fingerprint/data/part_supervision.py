from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .schema import PART_CLASSES


@dataclass(frozen=True)
class PartTargetRecord:
    label_path: Path
    class_supervision: np.ndarray
    source: str = "carparts"


def _parse_yolo_polygons(label_path: Path) -> list[tuple[int, np.ndarray]]:
    rows: list[tuple[int, np.ndarray]] = []
    if not label_path.is_file():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        toks = line.split()
        if len(toks) < 7:
            continue
        try:
            cid = int(float(toks[0]))
            coords = np.asarray([float(x) for x in toks[1:]], dtype=np.float32).reshape(-1, 2)
        except Exception:
            continue
        if cid < 0 or cid >= len(PART_CLASSES) or len(coords) < 3:
            continue
        rows.append((cid, np.clip(coords, 0.0, 1.0)))
    return rows


def rasterize_yolo_multilabel(
    label_path: str | Path,
    size: int | tuple[int, int] | list[int],
    *,
    horizontal_flip: bool = False,
) -> np.ndarray:
    if isinstance(size, int):
        H = W = int(size)
    else:
        H, W = int(size[0]), int(size[1])
    target = np.zeros((len(PART_CLASSES), H, W), dtype=np.uint8)
    for cid, xy in _parse_yolo_polygons(Path(label_path)):
        xy = xy.copy()
        if horizontal_flip:
            xy[:, 0] = 1.0 - xy[:, 0]
        pts = np.empty_like(xy, dtype=np.int32)
        pts[:, 0] = np.clip(np.rint(xy[:, 0] * (W - 1)), 0, W - 1).astype(np.int32)
        pts[:, 1] = np.clip(np.rint(xy[:, 1] * (H - 1)), 0, H - 1).astype(np.int32)
        cv2.fillPoly(target[cid], [pts], 1)
    return target


class YoloPartDatasetIndex:
    """Index the converted Carparts-Seg dataset."""
    def __init__(self, root: str | Path, split: str):
        root = Path(root)
        img_dir = root / "images" / split
        lab_dir = root / "labels" / split
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.items: list[tuple[Path, PartTargetRecord]] = []
        class_sup = np.ones(len(PART_CLASSES), dtype=np.float32)
        if not img_dir.exists() or not lab_dir.exists():
            return
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in exts:
                continue
            lab = lab_dir / f"{p.stem}.txt"
            if not lab.is_file():
                continue
            self.items.append((p, PartTargetRecord(lab, class_sup.copy())))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


class PartBootstrapDataset:
    """Carparts-Seg supervision for the DINO dense part head."""
    def __init__(
        self, root: str | Path, split: str, *,
        image_size: int | tuple[int, int] | list[int] = (256, 384),
        target_size: int | tuple[int, int] | list[int] = (64, 96),
        train: bool = True,
    ):
        from PIL import Image
        self.Image = Image
        self.index = YoloPartDatasetIndex(root, split)
        self.image_size = image_size
        self.target_size = target_size
        self.train = bool(train)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        from .dataset import _augment_image
        image_path, rec = self.index[idx]
        with self.Image.open(image_path) as im:
            image = im.convert("RGB")
        flip = bool(self.train and np.random.random() < 0.5)
        # Carparts bootstrap intentionally avoids geometric transforms other than a synced flip,
        # because polygon targets must remain pixel-aligned with the DINO feature grid.
        x, _ = _augment_image(
            image, self.image_size, self.train, clahe_prob=0.05 if self.train else 0.0,
            flip=flip, profile="part_bootstrap",
        )
        target = rasterize_yolo_multilabel(rec.label_path, self.target_size, horizontal_flip=flip)
        return {
            "image": x,
            "part_target": torch.from_numpy(target),
            "part_class_supervision": torch.from_numpy(rec.class_supervision.astype(np.float32)),
            "part_available": torch.tensor(1.0, dtype=torch.float32),
            "part_quality": torch.tensor(1.0, dtype=torch.float32),
            "part_negative_weight": torch.tensor(1.0, dtype=torch.float32),
            "path": str(image_path),
        }
