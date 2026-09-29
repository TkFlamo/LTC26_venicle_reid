from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "30_full_cv_pipeline.py"
    spec = importlib.util.spec_from_file_location("full_cv_pipeline_v094_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_optional_checkpoint_normalizes_pandas_nan_and_strings():
    m = _load_pipeline_module()
    assert m._optional_checkpoint(None) is None
    assert m._optional_checkpoint(np.nan) is None
    assert m._optional_checkpoint(float("nan")) is None
    assert m._optional_checkpoint("") is None
    assert m._optional_checkpoint("nan") is None
    assert m._optional_checkpoint("None") is None
    assert m._optional_checkpoint(" /tmp/model.pt ") == "/tmp/model.pt"


def test_direct_final_refit_cannot_become_transfer_from_nan(tmp_path, monkeypatch):
    m = _load_pipeline_module()
    commands = []

    def fake_run_cmd(cmd, **kwargs):
        commands.append([str(x) for x in cmd])
        marker = kwargs.get("marker")
        if marker:
            marker = Path(marker)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
        return True

    monkeypatch.setattr(m, "run_cmd", fake_run_cmd)
    pipe = {
        "paths": {"raw_images": "raw/images"},
        "resolutions": [[384, 576]],
        "baseline": {"profile": "base", "train_microbatch": {"vit_base": 2, "default": 4}},
        "device": "0",
        "precision": "bf16",
    }
    args = type("A", (), {"resume": True, "dry": False, "keep_going": False})()
    ok, _ = m._run_v5_exact_baseline(
        pipe, args, backbone="vit_base", train_manifest="dev.csv", val_manifest=None,
        run_dir=tmp_path / "run", epochs=16, final_refit=True, init_checkpoint=np.nan,
    )
    assert ok
    train_cmd = commands[0]
    assert "scripts/10_train_baseline_v5_exact.py" in train_cmd
    assert "scripts/10b_train_baseline_v5_transfer.py" not in train_cmd
    assert "--init-checkpoint" not in train_cmd
