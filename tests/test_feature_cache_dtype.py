from pathlib import Path
import subprocess
import sys

import numpy as np

from vehicle_fingerprint.features import _string_array, _assert_pickle_free_npz


def test_string_array_never_object():
    a = _string_array(['a', 2, None, np.str_('c')])
    assert a.dtype.kind in {'U', 'S'}
    assert a.tolist() == ['a', '2', 'None', 'c']


def test_pickle_free_check_accepts_unicode(tmp_path: Path):
    p = tmp_path / 'ok.npz'
    np.savez_compressed(p, vehicle_key=_string_array(['d:1', 'd:2']), z_fused=np.zeros((2, 4), np.float32))
    _assert_pickle_free_npz(p)


def test_repair_legacy_object_metadata(tmp_path: Path):
    src = tmp_path / 'legacy.npz'
    dst = tmp_path / 'fixed.npz'
    np.savez_compressed(src, vehicle_key=np.asarray(['d:1', 'd:2'], dtype=object), z_fused=np.zeros((2, 4), np.float32))
    script = Path(__file__).parents[1] / 'scripts' / '11b_repair_feature_cache.py'
    subprocess.run([sys.executable, str(script), '--input', str(src), '--out', str(dst), '--allow-legacy-pickle'], check=True)
    with np.load(dst, allow_pickle=False) as d:
        assert d['vehicle_key'].dtype.kind in {'U', 'S'}
        assert d['vehicle_key'].tolist() == ['d:1', 'd:2']
