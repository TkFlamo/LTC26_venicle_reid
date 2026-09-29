from __future__ import annotations

from pathlib import Path

from vehicle_fingerprint.external.importers import discover_veri776_root, import_veri776


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image-bytes")


def test_kaggle_nested_veri_layout_is_discovered(tmp_path: Path):
    root = tmp_path / "kaggle-download" / "some-wrapper" / "VeRi_with_plate"
    _touch(root / "image_train" / "0002_c002_00030600_0.jpg")
    _touch(root / "image_train" / "0002_c003_00030601_0.jpg")
    _touch(root / "image_train" / "0010_c001_00010000_0.png")
    _touch(root / "image_query" / "0002_c004_00040000_0.jpg")
    _touch(root / "image_test" / "0002_c005_00050000_0.jpg")

    resolved = discover_veri776_root(tmp_path / "kaggle-download")
    assert resolved == root.resolve()

    out = tmp_path / "veri776.csv"
    df = import_veri776(tmp_path / "kaggle-download", out)
    assert out.exists()
    assert set(df["split"]) == {"train", "query", "gallery"}
    assert len(df[df.split == "train"]) == 3
    assert df[df.split == "train"]["vehicle_id"].nunique() == 2
    assert df[df.split == "train"]["camera_id"].nunique() == 3
    assert set(df["vehicle_key"]) >= {"veri776:0002", "veri776:0010"}
    assert df.attrs["veri776_import_report"]["resolved_root"] == str(root.resolve())


def test_veri_direct_layout_and_aliases(tmp_path: Path):
    root = tmp_path / "VeRi"
    _touch(root / "bounding_box_train" / "0001_c001_x.jpg")
    _touch(root / "query" / "0001_c002_q.jpg")
    _touch(root / "bounding_box_test" / "0001_c003_g.jpg")

    df = import_veri776(root, tmp_path / "direct.csv")
    assert list(df.groupby("split").size().sort_index().items()) == [
        ("gallery", 1), ("query", 1), ("train", 1)
    ]
    assert df[df.split == "train"].iloc[0].camera_id == "001"
