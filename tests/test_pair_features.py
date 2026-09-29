import numpy as np
from vehicle_fingerprint.pairs import pair_feature_np


def test_pair_feature_shape():
    n=2;p=14;l=24;d=8
    c={
      'z_fused':np.array([[1,0],[1,0]],np.float32),
      'z_global':np.array([[1,0],[1,0]],np.float32),
      'parts':np.ones((n,p,d),np.float32)/np.sqrt(d),
      'visibility':np.ones((n,p),np.uint8),
      'local':np.ones((n,l,d),np.float32)/np.sqrt(d),
      'color':np.zeros((n,12),np.float32),
    }
    x=pair_feature_np(c,0,1)
    assert x.shape==(11+2*p,)
    assert x[0]>.99


def test_pair_feature_shape_with_v05_local_summary():
    n=2;p=14;l=24;d=8
    c={
      'z_fused':np.array([[1,0],[1,0]],np.float32),
      'z_global':np.array([[1,0],[1,0]],np.float32),
      'z_local':np.array([[1,0],[1,0]],np.float32),
      'parts':np.ones((n,p,d),np.float32)/np.sqrt(d),
      'visibility':np.ones((n,p),np.uint8),
      'local':np.ones((n,l,d),np.float32)/np.sqrt(d),
      'color':np.zeros((n,12),np.float32),
    }
    x=pair_feature_np(c,0,1)
    assert x.shape==(11+2*p,)
