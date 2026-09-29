from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from vehicle_fingerprint.data.color import robust_color_descriptor


def _reference_old(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > 18) & (gray < 242)
    pix = lab[mask]
    if len(pix) < 64:
        pix = lab.reshape(-1, 3)
    med = np.median(pix, axis=0)
    mad = np.median(np.abs(pix - med), axis=0)
    q25 = np.percentile(pix, 25, axis=0)
    q75 = np.percentile(pix, 75, axis=0)
    desc = np.concatenate([med, mad, q25, q75]).astype(np.float32)
    scale = np.tile(np.array([255.0, 255.0, 255.0], dtype=np.float32), 4)
    return desc / scale


def test_color_descriptor_is_bit_exact_to_previous_quantile_implementation():
    rng = np.random.default_rng(123)
    for h, w in [(64, 96), (192, 288), (384, 576)]:
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        np.testing.assert_array_equal(robust_color_descriptor(image), _reference_old(image))
