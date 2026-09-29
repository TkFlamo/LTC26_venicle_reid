from pathlib import Path
import csv
import numpy as np
import pandas as pd

from vehicle_fingerprint.data.official_protocol import build_official_protocol
from vehicle_fingerprint.official_eval import official_metrics_from_embeddings, load_official_evaluator
from vehicle_fingerprint.retrieval import run_retrieval


def test_protocol_and_official_metrics(tmp_path):
    rows=[]
    for vid in ['a','b','c','d']:
        for cam in ['1','2']:
            rows.append({'image_id':f'{vid}_{cam}','vehicle_id':vid,'camera_id':cam,'path':'/dev/null','dataset':'hackathon'})
    mf=tmp_path/'val.csv'; pd.DataFrame(rows).to_csv(mf,index=False)
    rep=build_official_protocol(mf,tmp_path/'proto',seed=1,open_set_fraction=.25,max_queries_per_id=1)
    assert rep['query_rows']>0 and rep['gallery_rows']>0 and rep['open_set_query_rows']>0
    q=pd.read_csv(tmp_path/'proto/query.csv'); g=pd.read_csv(tmp_path/'proto/gallery.csv')
    # Identity one-hot makes every known query retrieve its correct gallery identity first.
    all_ids=sorted(set(q.vehicle_id.astype(str))|set(g.vehicle_id.astype(str))); mp={x:i for i,x in enumerate(all_ids)}
    qe=np.eye(len(all_ids),dtype=np.float32)[[mp[x] for x in q.vehicle_id.astype(str)]]
    ge=np.eye(len(all_ids),dtype=np.float32)[[mp[x] for x in g.vehicle_id.astype(str)]]
    r=official_metrics_from_embeddings(qe,ge,q_ids=q.image_id.astype(str).tolist(),g_ids=g.image_id.astype(str).tolist(),gt_csv=tmp_path/'proto/ground_truth.csv')
    assert r['ranking']['mAP@10']==1.0
    assert r['ranking']['Rank-1']==1.0


def test_official_junk_is_same_id_same_camera_only(tmp_path):
    gt=pd.DataFrame([
        {'image_id':'q','vehicle_id':'A','camera_id':'1','split':'query'},
        {'image_id':'junk','vehicle_id':'A','camera_id':'1','split':'gallery'},
        {'image_id':'neg_same_cam','vehicle_id':'B','camera_id':'1','split':'gallery'},
        {'image_id':'pos','vehicle_id':'A','camera_id':'2','split':'gallery'},
    ])
    p=tmp_path/'gt.csv';gt.to_csv(p,index=False)
    mod=load_official_evaluator()
    q,g=mod.load_gt(str(p))
    clean=mod.strip_junk(['junk','neg_same_cam','pos'],q.loc['q'],g.vehicle_id.to_dict(),g.camera_id.to_dict())
    assert clean==['neg_same_cam','pos']


def test_retrieval_writes_official_artifacts(tmp_path):
    q={'z_global':np.array([[1.,0.]],np.float32),'z_fused':np.array([[1.,0.]],np.float32),'sample_id':np.array(['sq']), 'meta_image_id':np.array(['q'])}
    g={'z_global':np.array([[1.,0.],[0.,1.]],np.float32),'z_fused':np.array([[1.,0.],[0.,1.]],np.float32),'sample_id':np.array(['sg1','sg2']), 'meta_image_id':np.array(['g1','g2'])}
    qp=tmp_path/'q.npz';gp=tmp_path/'g.npz';np.savez(qp,**q);np.savez(gp,**g)
    out=run_retrieval(qp,gp,tmp_path/'out',output_topk=2)
    with open(out/'submission.csv',newline='',encoding='utf-8') as f:
        rows=list(csv.reader(f))
    assert rows==[['q','g1','g2']]
    c=pd.read_csv(out/'candidates.csv'); assert list(c.columns)==['query_id','gallery_id','confidence']; assert c.iloc[0].gallery_id=='g1'
    e=np.load(out/'embeddings.npy'); assert e.shape==(3,2)
