import numpy as np
from vehicle_fingerprint.score_ensemble import dense_final_scores, fuse_score_matrices, ranked_dict_from_matrix, weighted_concat_embeddings, zscore_rows


def test_dense_and_zscore_fusion_matches_manual():
    qids=["q1","q2"]; gids=["g1","g2","g3"]
    r1={"q1":["g1","g2","g3"],"q2":["g2","g1","g3"]}
    s1={"q1":np.array([.9,.4,.1]),"q2":np.array([.8,.5,.2])}
    r2={"q1":["g2","g1","g3"],"q2":["g1","g3","g2"]}
    s2={"q1":np.array([.7,.6,.2]),"q2":np.array([.9,.3,.1])}
    a=dense_final_scores(r1,s1,qids,gids); b=dense_final_scores(r2,s2,qids,gids)
    got=fuse_score_matrices([a,b],[.6,.4])
    exp=.6*zscore_rows(a)+.4*zscore_rows(b)
    np.testing.assert_allclose(got,exp,rtol=0,atol=1e-7)
    ranked=ranked_dict_from_matrix(got,qids,gids)
    assert set(ranked)==set(qids)
    assert all(len(v)==3 for v in ranked.values())


def test_weighted_concat_cosine_matches_weighted_member_cosines():
    a=np.array([[1.,0.],[0.,1.]],np.float32)
    b=np.array([[1.,1.],[1.,-1.]],np.float32)
    w=[.65,.35]
    z=weighted_concat_embeddings([a,b],w)
    an=a/np.linalg.norm(a,axis=1,keepdims=True)
    bn=b/np.linalg.norm(b,axis=1,keepdims=True)
    expected=.65*(an@an.T)+.35*(bn@bn.T)
    np.testing.assert_allclose(z@z.T,expected,atol=1e-6,rtol=0)
    np.testing.assert_allclose(np.linalg.norm(z,axis=1),1.0,atol=1e-6)
