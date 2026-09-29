from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "21_infer_from_csv.py"


def test_infer_script_has_safe_main_guard():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in text
    assert "freeze_support()" in text
    assert 'sys.path.insert(0, str(SRC))' in text


def test_local_src_wins_over_inherited_pythonpath(tmp_path):
    wrong = tmp_path / "vehicle_fingerprint"
    wrong.mkdir()
    (wrong / "__init__.py").write_text('raise RuntimeError("wrong package")\n', encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    p = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert p.returncode == 0, p.stdout
    assert "--workers" in p.stdout


def test_mp_child_import_does_not_execute_inference(monkeypatch):
    # Windows spawn imports the main module under __mp_main__.  The script must
    # define functions/imports only and must not parse CLI arguments or launch
    # feature extraction in that mode.
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    ns = runpy.run_path(str(SCRIPT), run_name="__mp_main__")
    assert callable(ns["main"])
    assert ns["ROOT"] == ROOT
