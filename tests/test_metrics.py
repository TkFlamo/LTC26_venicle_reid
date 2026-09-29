import numpy as np
from vehicle_fingerprint.metrics import cross_camera_retrieval_metrics, refusal_metrics


def test_cross_camera_metrics_perfect():
    z=np.array([[1,0],[1,0],[0,1],[0,1]],np.float32)
    vid=np.array(['a','a','b','b']); cam=np.array(['1','2','1','2'])
    m=cross_camera_retrieval_metrics(z,vid,cam)
    assert m['mAP'] > .99
    assert m['Rank-1'] > .99
    assert m['Rank-5'] > .99
    assert m['mINP'] > .99


def test_refusal_metrics_pr_auc():
    y=np.array([1,1,0,0])
    p=np.array([.95,.8,.2,.05])
    m=refusal_metrics(y,p,.5)
    assert m['F1'] > .99
    assert m['TNR'] > .99
    assert m['PR-AUC'] > .99
