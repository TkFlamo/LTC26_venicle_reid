from __future__ import annotations

from pathlib import Path
from PIL import Image


def find_image(images_dir: str | Path, image_id: str) -> Path:
    root = Path(images_dir)
    raw = str(image_id)
    # Accept both organizer-style bare IDs and defensive IDs that already include an image extension.
    direct = root / raw
    if direct.is_file():
        return direct
    stem = Path(raw).stem if Path(raw).suffix else raw
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = root / f"{stem}{ext}"
        if p.is_file():
            return p
    # Fallback for nested image folders.
    hits = []
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        hits.extend(root.rglob(f"{stem}.{ext}"))
        if hits:
            break
    if not hits:
        raise FileNotFoundError(f"Image {image_id!r} not found under {root}")
    return hits[0]


def bbox_crop_bounds(
    image_size: tuple[int, int],
    x: float,
    y: float,
    w: float,
    h: float,
    pad: float = 0.08,
) -> tuple[int, int, int, int]:
    """Return the exact clipped crop bounds used by :func:`crop_bbox`.

    Keeping this geometry explicit makes preprocessing reproducible across training,
    feature extraction and inference. The organizer bbox may be loose; downstream code can
    still reconstruct exactly the same padded crop even at image borders.
    """
    W, H = map(int, image_size)
    px, py = float(w) * float(pad), float(h) * float(pad)
    x0 = max(0, int(round(float(x) - px)))
    y0 = max(0, int(round(float(y) - py)))
    x1 = min(W, int(round(float(x) + float(w) + px)))
    y1 = min(H, int(round(float(y) + float(h) + py)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid clipped bbox {(x, y, w, h)} for image size {(W, H)}")
    return x0, y0, x1, y1


def target_bbox_in_crop(
    image_size: tuple[int, int],
    x: float,
    y: float,
    w: float,
    h: float,
    pad: float = 0.08,
) -> dict[str, float | int]:
    """Geometry of the organizer bbox in the materialized padded crop.

    The target bbox is not assumed to be a perfect segmentation. The geometry is retained
    for reproducibility and diagnostics; ReID preprocessing uses the original bbox plus padding.
    """
    x0, y0, x1, y1 = bbox_crop_bounds(image_size, x, y, w, h, pad)
    cw, ch = x1 - x0, y1 - y0
    tx0 = max(0.0, float(x) - x0)
    ty0 = max(0.0, float(y) - y0)
    tx1 = min(float(cw), float(x) + float(w) - x0)
    ty1 = min(float(ch), float(y) + float(h) - y0)
    tcx = 0.5 * (tx0 + tx1)
    tcy = 0.5 * (ty0 + ty1)
    return {
        "crop_x0": int(x0),
        "crop_y0": int(y0),
        "crop_x1": int(x1),
        "crop_y1": int(y1),
        "crop_w": int(cw),
        "crop_h": int(ch),
        "target_x0_crop": float(tx0),
        "target_y0_crop": float(ty0),
        "target_x1_crop": float(tx1),
        "target_y1_crop": float(ty1),
        "target_cx_crop": float(tcx),
        "target_cy_crop": float(tcy),
        "target_cx_norm": float(tcx / max(1, cw)),
        "target_cy_norm": float(tcy / max(1, ch)),
    }


def crop_bbox(image: Image.Image, x: float, y: float, w: float, h: float, pad: float = 0.08) -> Image.Image:
    x0, y0, x1, y1 = bbox_crop_bounds(image.size, x, y, w, h, pad)
    return image.crop((x0, y0, x1, y1))
