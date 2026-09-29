from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.baseline_v5_exact import exact_v5_base_argv


def _value(argv, key):
    idx = [i for i, x in enumerate(argv) if x == key][-1]
    return argv[idx + 1]


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_exact_v5_exposes_transfer_init_without_changing_profile():
    argv = exact_v5_base_argv(
        model_name="convnext_base.dinov3_lvd1689m",
        height=384, width=576, out="x", images_dir="veri", train_csv="train.csv",
        device="0", init_checkpoint="veri_best.pt", plate_mask_prob=0.35,
    )
    assert _value(argv, "--init-checkpoint") == "veri_best.pt"
    assert float(_value(argv, "--plate-mask-prob")) == 0.35
    assert (_value(argv, "--p"), _value(argv, "--k")) == ("16", "4")
    assert _value(argv, "--lr-backbone") == "3e-5"
    assert _value(argv, "--lr-head") == "3e-4"


def test_veri_prepare_writes_v5_relative_path_manifests(tmp_path: Path):
    veri = tmp_path / "VeRi"
    _touch(veri / "image_train" / "0001_c001_a.jpg")
    _touch(veri / "image_train" / "0001_c002_b.jpg")
    _touch(veri / "image_train" / "0002_c001_c.jpg")
    _touch(veri / "image_query" / "1001_c003_q.jpg")
    _touch(veri / "image_test" / "1001_c004_g.jpg")
    out = tmp_path / "processed"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "09_prepare_external_reid.py"),
        "--veri-root", str(veri), "--out-dir", str(out), "--root", str(tmp_path)
    ], check=True, cwd=ROOT, env=env)
    tr = (out / "veri_v5_train.csv").read_text()
    va = (out / "veri_v5_val.csv").read_text()
    assert "image_train/0001_c001_a.jpg" in tr
    assert "image_query/1001_c003_q.jpg" in va
    assert "image_test/1001_c004_g.jpg" in va
    meta = json.loads((out / "veri_v5_transfer_manifest.json").read_text())
    assert meta["train_val_identity_overlap"] == 0
    assert meta["train_ids"] == 2 and meta["val_ids"] == 1


def test_clean_v094_configs_support_direct_default_and_optional_veri():
    direct = yaml.safe_load((ROOT / "configs" / "full_cv_pipeline.yaml").read_text())
    veri = yaml.safe_load((ROOT / "configs" / "full_cv_pipeline_with_veri.yaml").read_text())
    assert direct["veri_pretrain"]["enabled"] is False
    assert veri["veri_pretrain"]["enabled"] is True
    assert veri["veri_pretrain"]["run_direct_target_control"] is True
    assert veri["veri_pretrain"]["train_manifest"].endswith("veri_v5_train.csv")
    assert veri["veri_pretrain"]["val_manifest"].endswith("veri_v5_val.csv")
    for cfg in (direct, veri):
        assert cfg["cv"]["shared_validation_spec"] == "configs/shared_validation_v5.json"
        assert cfg["backbones"] == ["convnext_base", "vit_base"]
        assert cfg["resolutions"] == [[384, 576]]


def test_v5_cli_exposes_transfer_arguments():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    p = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "10_train_baseline_v5_exact.py"), "--help"],
        check=True, cwd=ROOT, env=env, text=True, capture_output=True,
    )
    assert "--init-checkpoint" in p.stdout
    assert "--plate-mask-prob" in p.stdout
