from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_vendored_external_convnext_trainer_is_present():
    root = ROOT / "third_party" / "external_convnext_v5_training"
    for rel in (
        "vehicle_reid_v5.py",
        "train_best_v5.py",
        "run_next_experiments_v7.py",
        "requirements_v5.txt",
        "README_INTEGRATION.md",
        "SOURCE_SHA256.txt",
        "reference/E3e_camera_strongerase_history.csv",
        "reference/E3e_camera_strongerase_split.json",
    ):
        assert (root / rel).is_file(), rel


def test_external_convnext_wrapper_targets_winning_profile():
    mod = _load("ext_train_wrapper", ROOT / "scripts" / "67_train_external_convnext_best_map_v094.py")
    assert mod.PROFILE == "small_camera_sampler_strongerase"
    assert mod.REFERENCE_BEST_EPOCH == 25
    assert abs(mod.REFERENCE_MAP10 - 0.8214526158811873) < 1e-12
