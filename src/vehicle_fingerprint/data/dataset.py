from __future__ import annotations

import math
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .schema import PART_CLASSES
from .color import robust_color_descriptor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_MEAN_T = torch.tensor(IMAGENET_MEAN)[:, None, None]
_STD_T = torch.tensor(IMAGENET_STD)[:, None, None]


def _hw(size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(size, int):
        return int(size), int(size)
    return int(size[0]), int(size[1])


def _clahe_rgb(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def _crop_source_bbox(image: Image.Image, row, pad_frac: float) -> Image.Image:
    W, H = image.size
    x, y, w, h = map(float, (row.x, row.y, row.w, row.h))
    px, py = w * float(pad_frac), h * float(pad_frac)
    x1 = max(0, int(math.floor(x - px))); y1 = max(0, int(math.floor(y - py)))
    x2 = min(W, int(math.ceil(x + w + px))); y2 = min(H, int(math.ceil(y + h + py)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid source bbox {(x1, y1, x2, y2)}")
    return image.crop((x1, y1, x2, y2))


def _baseline_v4_transform(size, train: bool):
    h, w = _hw(size)
    resize = transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC, antialias=True)
    if not train:
        return transforms.Compose([
            resize,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        resize,
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.ColorJitter(0.16, 0.16, 0.10, 0.02)], p=0.65),
        transforms.RandomAffine(
            3.0, translate=(0.02, 0.02), scale=(0.97, 1.03), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.22, scale=(0.015, 0.08), ratio=(0.4, 2.5), value="random"),
    ])


def _augment_image(
    image: Image.Image,
    size: int | tuple[int, int] | list[int],
    train: bool,
    clahe_prob: float,
    *,
    flip: bool | None = None,
    profile: str = "part_bootstrap",
):
    """Light augmentation used for Carparts-Seg where masks must stay aligned.

    The strong ReID baseline uses :func:`_baseline_v4_transform` instead.
    """
    h, w = _hw(size)
    arr = np.asarray(image.convert("RGB"))
    if flip is None:
        flip = bool(train and random.random() < 0.5)
    if flip:
        arr = np.ascontiguousarray(arr[:, ::-1])
    if train:
        pil = Image.fromarray(arr)
        if random.random() < 0.55:
            pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.85, 1.15))
            pil = ImageEnhance.Contrast(pil).enhance(random.uniform(0.86, 1.14))
            pil = ImageEnhance.Color(pil).enhance(random.uniform(0.90, 1.10))
        arr = np.asarray(pil)
        if random.random() < float(clahe_prob):
            arr = _clahe_rgb(arr)
    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA if max(arr.shape[:2]) > max(h, w) else cv2.INTER_CUBIC)
    x = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
    x = (x - _MEAN_T) / _STD_T
    return x, bool(flip)


class ReIDDataset(Dataset):
    """Hackathon identity dataset with baseline-faithful crop/preprocessing.

    Semantic parts are learned only from Carparts-Seg and are predicted at runtime from
    the same DINOv3/ConvNeXt dense features.
    """
    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        *,
        image_size: int | tuple[int, int] | list[int] = (256, 384),
        train: bool = True,
        label_map: dict[str, int] | None = None,
        augmentation_profile: str = "baseline_v4",
        use_source_bbox: bool = True,
        bbox_pad: float = 0.03,
        part_target_size: int | tuple[int, int] | list[int] = (64, 96),
        clahe_prob: float = 0.0,
        return_weak_view: bool = False,
        inference_only: bool = False,
        return_color_descriptor: bool = False,
        **_legacy_unused,
    ):
        self.df = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.reset_index(drop=True).copy()
        self.image_size = image_size
        self.train = bool(train)
        self.augmentation_profile = str(augmentation_profile)
        self.use_source_bbox = bool(use_source_bbox)
        self.bbox_pad = float(bbox_pad)
        self.part_target_size = _hw(part_target_size)
        self.return_weak_view = bool(return_weak_view)
        self.inference_only = bool(inference_only)
        self.return_color_descriptor = bool(return_color_descriptor)
        if "dataset" not in self.df.columns:
            self.df["dataset"] = "hackathon"
        if "camera_id" not in self.df.columns:
            self.df["camera_id"] = "-1"
        keys = (self.df["dataset"].astype(str) + ":" + self.df["vehicle_id"].astype(str)).tolist()
        if label_map is None:
            label_map = {x: i for i, x in enumerate(sorted(set(keys)))}
        self.label_map = label_map
        self.keys = keys
        self.transform = _baseline_v4_transform(image_size, self.train) if self.augmentation_profile == "baseline_v4" else None
        self.weak_transform = _baseline_v4_transform(image_size, False)

    def __len__(self):
        return len(self.df)

    def _open_crop(self, row) -> Image.Image:
        can_source = (
            self.use_source_bbox
            and "source_path" in row.index and Path(str(row.source_path)).is_file()
            and all(k in row.index for k in ("x", "y", "w", "h"))
        )
        path = str(row.source_path) if can_source else str(row.path)
        with Image.open(path) as im:
            image = im.convert("RGB")
            if can_source:
                image = _crop_source_bbox(image, row, self.bbox_pad)
            return image.copy()

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sid = str(row.sample_id) if "sample_id" in row.index else str(row.image_id)
        image = self._open_crop(row)
        if self.transform is not None:
            x = self.transform(image)
        else:
            x, _ = _augment_image(image, self.image_size, self.train, 0.0, profile="part_bootstrap")
        out = {
            "image": x,
            "vehicle_key": self.keys[idx],
            "camera_id": str(row.camera_id),
            "sample_id": sid,
            "path": str(row.source_path) if self.use_source_bbox and "source_path" in row.index else str(row.path),
            "orig_size": image.size[::-1],
        }
        if self.return_color_descriptor:
            # Compute exactly the same descriptor as the legacy main-process path, but inside
            # DataLoader workers so CPU colour statistics overlap the GPU forward pass.
            rgb = (x * _STD_T + _MEAN_T).clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
            out["color"] = torch.from_numpy(robust_color_descriptor(Image.fromarray(rgb), None))
        if not self.inference_only:
            ph, pw = self.part_target_size
            out.update({
                "label": torch.tensor(self.label_map[self.keys[idx]], dtype=torch.long),
                "part_target": torch.zeros((len(PART_CLASSES), ph, pw), dtype=torch.uint8),
                "part_class_supervision": torch.zeros(len(PART_CLASSES), dtype=torch.float32),
                "part_available": torch.tensor(0.0, dtype=torch.float32),
                "part_quality": torch.tensor(0.0, dtype=torch.float32),
                "part_negative_weight": torch.tensor(0.0, dtype=torch.float32),
            })
        if self.return_weak_view:
            out["weak_image"] = self.weak_transform(image)
        return out
