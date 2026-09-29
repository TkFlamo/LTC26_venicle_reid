from __future__ import annotations

from pathlib import Path
import urllib.request
import zipfile

from ..utils import load_yaml


def download_external(config: str | Path, root: str | Path, names: list[str] | None = None) -> None:
    cfg = load_yaml(config)["datasets"]
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    selected = names or [k for k, v in cfg.items() if v.get("enabled", True)]
    for name in selected:
        if name not in cfg:
            raise KeyError(f"Unknown external dataset {name!r}. Available: {', '.join(cfg)}")
        spec = cfg[name]
        target = root / spec["expected_dir"]
        if target.exists():
            print(f"[ok] {name}: {target}")
            continue
        if spec.get("access") != "automatic":
            print(f"[manual] {name}")
            if spec.get("download"):
                print(f"  download: {spec['download']}")
            if spec.get("official"):
                print(f"  original project: {spec['official']}")
            print(f"  place/extract anywhere under: {target}")
            if name == "veri776":
                print("  Kaggle wrapper folders are supported automatically; image_train is discovered recursively.")
            continue
        url = spec["url"]
        archive = root / f".{name}.zip"
        print(f"[download] {name}: {url}")
        urllib.request.urlretrieve(url, archive)
        print(f"[extract] {archive}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
        archive.unlink(missing_ok=True)
        if not target.exists():
            print(f"[warn] extracted but expected {target} was not found; inspect {root}")
