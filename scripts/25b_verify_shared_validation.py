#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import pandas as pd


def sha(vals):
    return hashlib.sha256("\n".join(map(str, vals)).encode()).hexdigest()

p=argparse.ArgumentParser(description='Verify the fixed cross-project vehicle_reid_v5 shared validation split and protocol fingerprints')
p.add_argument('--cv-root',default='data/processed/hackathon_single_v5shared')
p.add_argument('--spec',default='configs/shared_validation_v5.json')
a=p.parse_args()
root=Path(a.cv_root);spec=json.loads(Path(a.spec).read_text(encoding='utf-8'))
va=pd.read_csv(root/'shared_val.csv',dtype={'image_id':str,'vehicle_id':str,'camera_id':str})
checks={
 'rows':len(va),
 'vehicle_ids':va.vehicle_id.nunique(),
 'cameras':va.camera_id.nunique(),
 'vehicle_ids_sha256':sha(sorted(va.vehicle_id.astype(str).unique())),
 'image_ids_sorted_sha256':sha(sorted(va.image_id.astype(str).tolist())),
 'image_ids_ordered_sha256':sha(va.image_id.astype(str).tolist()),
}
exp=spec['dataset_expectations'];fps=spec['fingerprints']
assert checks['rows']==exp['shared_validation_rows']
assert checks['vehicle_ids']==exp['shared_validation_vehicle_ids']
assert checks['vehicle_ids_sha256']==fps['val_vehicle_ids_sha256']
assert checks['image_ids_sorted_sha256']==fps['val_image_ids_sorted_sha256']
assert checks['image_ids_ordered_sha256']==fps['val_image_ids_ordered_sha256']
folds=[]
for fexp in spec['official_protocol']['fold_fingerprints']:
    f=int(fexp['fold']);fd=root/'shared_validation'/f'fold_{f}'
    q=pd.read_csv(fd/'query.csv',dtype={'image_id':str});g=pd.read_csv(fd/'gallery.csv',dtype={'image_id':str})
    got={'fold':f,'query_rows':len(q),'gallery_rows':len(g),'query_sha256':sha(q.image_id.tolist()),'gallery_sha256':sha(g.image_id.tolist())}
    assert got['query_sha256']==fexp['query_image_ids_sha256']
    assert got['gallery_sha256']==fexp['gallery_image_ids_sha256']
    folds.append(got)
print(json.dumps({'status':'OK','shared_validation':checks,'protocol_folds':folds},ensure_ascii=False,indent=2))
