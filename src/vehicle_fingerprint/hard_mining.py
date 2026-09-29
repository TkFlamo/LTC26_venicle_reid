from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from .features import load_cache


def _normalize(x):
    x=x.astype(np.float32,copy=True); x/=np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12); return x


def _score_embedding(c, representation: str, alpha: float):
    zg=_normalize(c["z_global"]); zf=_normalize(c["z_fused"])
    if representation=="global": return zg
    if representation=="fused": return zf
    if representation=="blend":
        a=float(np.clip(alpha,0,1))
        # Concatenation makes dot product equal the convex score blend.
        return np.concatenate([np.sqrt(1-a)*zg,np.sqrt(a)*zf],axis=1).astype(np.float32)
    raise ValueError(f"Unknown representation={representation!r}")


def mine_identity_hard_negatives(cache_path: str | Path, out_json: str | Path, topk: int = 40,
                                 representation: str = "blend", alpha: float = 0.25,
                                 refine_factor: int = 4, top_pair_mean: int = 3):
    """Mine identity hard negatives, refining prototype neighbours with view-level similarities."""
    c=load_cache(cache_path); z=_score_embedding(c,representation,alpha); keys=c["vehicle_key"].astype(str)
    uniq=sorted(set(keys)); by={k:np.flatnonzero(keys==k) for k in uniq}; prot=[]
    for k in uniq:
        v=z[by[k]].mean(0); v/=max(1e-12,np.linalg.norm(v)); prot.append(v)
    prot=np.stack(prot).astype(np.float32)
    nprobe=min(len(uniq),max(topk+1,int(topk*max(1,refine_factor))+1))
    try:
        import faiss
        index=faiss.IndexFlatIP(prot.shape[1]); index.add(prot); _,I=index.search(prot,nprobe)
    except Exception:
        I=np.argsort(-(prot@prot.T),axis=1)[:,:nprobe]
    out={}
    m=max(1,int(top_pair_mean))
    for i,k in enumerate(uniq):
        cand=[int(j) for j in I[i] if int(j)!=i]
        scored=[]; aidx=by[k]
        for j in cand:
            bidx=by[uniq[j]]
            ps=(z[aidx]@z[bidx].T).reshape(-1)
            take=min(m,len(ps))
            score=float(np.partition(ps,-take)[-take:].mean()) if take else -1.0
            scored.append((score,uniq[j]))
        scored.sort(reverse=True)
        out[k]=[name for _,name in scored[:topk]]
    Path(out_json).parent.mkdir(parents=True,exist_ok=True)
    json.dump(out,open(out_json,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    return out


def mine_identity_hard_negatives_from_arrays(
    z: np.ndarray,
    keys,
    *,
    topk: int = 40,
    refine_factor: int = 4,
    top_pair_mean: int = 3,
) -> dict[str, list[str]]:
    """In-memory variant used for periodic online hard-negative refresh."""
    z=_normalize(np.asarray(z,np.float32)); keys=np.asarray(keys).astype(str)
    uniq=sorted(set(keys.tolist())); by={k:np.flatnonzero(keys==k) for k in uniq}; prot=[]
    for k in uniq:
        v=z[by[k]].mean(0); v/=max(1e-12,np.linalg.norm(v)); prot.append(v)
    prot=np.stack(prot).astype(np.float32)
    nprobe=min(len(uniq),max(topk+1,int(topk*max(1,refine_factor))+1))
    try:
        import faiss
        index=faiss.IndexFlatIP(prot.shape[1]); index.add(prot); _,I=index.search(prot,nprobe)
    except Exception:
        I=np.argsort(-(prot@prot.T),axis=1)[:,:nprobe]
    out={};m=max(1,int(top_pair_mean))
    for i,k in enumerate(uniq):
        scored=[];aidx=by[k]
        for j in [int(j) for j in I[i] if int(j)!=i]:
            ps=(z[aidx]@z[by[uniq[j]]].T).reshape(-1);take=min(m,len(ps))
            score=float(np.partition(ps,-take)[-take:].mean()) if take else -1.0
            scored.append((score,uniq[j]))
        scored.sort(reverse=True);out[k]=[name for _,name in scored[:topk]]
    return out
