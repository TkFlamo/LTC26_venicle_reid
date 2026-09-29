from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def robust_color_descriptor(image: Image.Image, foreground: np.ndarray | None = None) -> np.ndarray:
    """Robust Lab color descriptor: median, MAD and quartiles over likely paint pixels.

    This is deliberately non-generative: it estimates color evidence but never modifies the image.
    """
    rgb = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Remove very dark / saturated highlight pixels, which are dominated by reflections.
    mask = (gray > 18) & (gray < 242)
    if foreground is not None:
        fg = cv2.resize(foreground.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        mask &= fg
    pix = lab[mask]
    if len(pix) < 64:
        pix = lab.reshape(-1, 3)
    # Compute the three ordinary quantiles in one partition pass.  This is
    # numerically identical to the previous separate median/q25/q75 calls, but
    # materially cheaper on the 384x576 inference crop (the descriptor is one
    # of the dominant CPU costs when DataLoader workers are disabled).
    q25, med, q75 = np.percentile(pix, [25, 50, 75], axis=0)
    mad = np.median(np.abs(pix - med), axis=0)
    desc = np.concatenate([med, mad, q25, q75]).astype(np.float32)
    # Normalize each Lab-like block to stable scales.
    scale = np.tile(np.array([255.0, 255.0, 255.0], dtype=np.float32), 4)
    return desc / scale
