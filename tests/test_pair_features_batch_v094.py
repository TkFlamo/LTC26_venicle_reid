import numpy as np
from vehicle_fingerprint.pairs import pair_feature_np_cross, pair_features_np_cross_batch


def _norm_last(x):
    x=np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-6)


def test_pair_batch_matches_scalar():
    rng=np.random.default_rng(7); nq,ng=3,9; slots,local_n=6,24
    q={
        'z_fused':_norm_last(rng.normal(size=(nq,16))),
        'z_global':_norm_last(rng.normal(size=(nq,16))),
        'z_local':_norm_last(rng.normal(size=(nq,12))),
        'local':_norm_last(rng.normal(size=(nq,local_n,8))).astype(np.float16),
        'parts':_norm_last(rng.normal(size=(nq,slots,8))).astype(np.float16),
        'visibility':(rng.random((nq,slots))>.35).astype(np.uint8),
        'visibility_score':rng.random((nq,slots)).astype(np.float16),
        'color':rng.random((nq,18)).astype(np.float32),
        'camera_id':np.array(['1','2','1']),
    }
    g={
        'z_fused':_norm_last(rng.normal(size=(ng,16))),
        'z_global':_norm_last(rng.normal(size=(ng,16))),
        'z_local':_norm_last(rng.normal(size=(ng,12))),
        'local':_norm_last(rng.normal(size=(ng,local_n,8))).astype(np.float16),
        'parts':_norm_last(rng.normal(size=(ng,slots,8))).astype(np.float16),
        'visibility':(rng.random((ng,slots))>.35).astype(np.uint8),
        'visibility_score':rng.random((ng,slots)).astype(np.float16),
        'color':rng.random((ng,18)).astype(np.float32),
        'camera_id':np.array([str(x%3) for x in range(ng)]),
    }
    # Exercise no-visible-parts edge case too.
    q['visibility'][1,:]=0
    js=np.array([0,2,3,7,8]); jac=np.array([0,.1,.2,.3,.4],np.float32)
    for qi in range(nq):
        scalar=np.stack([pair_feature_np_cross(q,qi,g,int(j),neighbor_jaccard=float(z)) for j,z in zip(js,jac)])
        batch=pair_features_np_cross_batch(q,qi,g,js,neighbor_jaccard=jac)
        np.testing.assert_allclose(batch,scalar,rtol=2e-5,atol=2e-5)
