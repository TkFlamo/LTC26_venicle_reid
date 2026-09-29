from pathlib import Path
import hashlib, json
import pandas as pd

from vehicle_fingerprint.data.kfold import build_v5_shared_protocols


def _sha(vals):
    return hashlib.sha256("\n".join(map(str, vals)).encode()).hexdigest()


def test_packaged_v5_shared_validation_spec_is_self_consistent(tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'configs/shared_validation_v5.json').read_text(encoding='utf-8'))
    rows=spec['ordered_validation_rows']
    df=pd.DataFrame(rows)
    assert len(df)==1430
    assert df.vehicle_id.nunique()==231
    assert df.camera_id.nunique()==87
    assert _sha(sorted(df.vehicle_id.astype(str).unique()))==spec['fingerprints']['val_vehicle_ids_sha256']
    assert _sha(df.image_id.astype(str).tolist())==spec['fingerprints']['val_image_ids_ordered_sha256']
    rep=build_v5_shared_protocols(df,tmp_path/'shared_validation',spec)
    assert rep['folds']==4
    assert [x['query_rows'] for x in rep['reports']]==[231,231,231,231]
    assert [x['open_set_ids'] for x in rep['reports']]==[46,46,46,46]
    assert [x['gallery_rows'] for x in rep['reports']]==[968,954,955,966]
