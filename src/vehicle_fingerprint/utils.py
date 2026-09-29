from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(device: str | int | None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        return torch.device(f"cuda:{device}")
    return torch.device(str(device))


def autocast_context(device: torch.device, precision: str):
    precision = precision.lower()
    if device.type != "cuda" or precision in {"fp32", "float32", "32"}:
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision in {"bf16", "bfloat16"} else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)
