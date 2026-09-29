from __future__ import annotations

from pathlib import Path
import json, random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .features import load_cache
from .models.reranker import PairReranker


def _normalize(x):
    x=np.asarray(x,np.float32).copy();x/=np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12);return x


def build_same_cache_knn(cache: dict, k: int = 20) -> list[set[int]]:
    z=_normalize(cache["z_global"]);sim=z@z.T;np.fill_diagonal(sim,-np.inf)
    kk=min(max(1,int(k)),max(1,len(z)-1)); order=np.argsort(-sim,axis=1,kind="stable")[:,:kk]
    return [set(map(int,row.tolist())) for row in order]


def build_cross_knn_context(q: dict, g: dict, k: int = 20):
    """Return expanded k-reciprocal neighbourhoods in one shared query+gallery index space.

    This is a compact ReID-style approximation of k-reciprocal/Jaccard re-ranking.  A neighbour is
    kept only when the relation is mutual; reciprocal sets are then expanded with half-k reciprocal
    neighbours that substantially overlap the seed set.  The returned sets can be Jaccard-compared
    for both reranker features and score-level re-ranking.
    """
    qz=_normalize(q["z_global"]);gz=_normalize(g["z_global"]);z=np.concatenate([qz,gz],axis=0)
    n=len(z); nq=len(qz)
    if n<=1:return [set() for _ in range(nq)],[set() for _ in range(len(gz))]
    kk=min(max(1,int(k)),n-1); half=max(1,kk//2)
    sim=z@z.T;np.fill_diagonal(sim,-np.inf)
    order=np.argsort(-sim,axis=1,kind="stable")[:,:kk]
    half_order=order[:,:half]
    reciprocal=[]
    for i in range(n):
        fwd=order[i]; rs={int(j) for j in fwd if i in order[int(j)]}
        if not rs:rs=set(map(int,half_order[i].tolist()))
        reciprocal.append(rs)
    expanded=[]
    for i,base in enumerate(reciprocal):
        ex=set(base)
        for j in list(base):
            cand=reciprocal[int(j)]
            if not cand:continue
            overlap=len(base & cand)/max(1,len(cand))
            if overlap>=2.0/3.0:ex.update(cand)
        expanded.append(ex)
    return expanded[:nq],expanded[nq:]

def _jaccard(a:set[int],b:set[int])->float:
    u=len(a|b);return float(len(a&b)/u) if u else 0.0


def _part_evidence(c1,i:int,c2,j:int):
    p1=c1["parts"][i].astype(np.float32);p2=c2["parts"][j].astype(np.float32);ps=np.sum(p1*p2,axis=-1)
    v1=c1["visibility"][i].astype(bool);v2=c2["visibility"][j].astype(bool);joint=v1&v2
    if "visibility_score" in c1 and "visibility_score" in c2:
        conf=np.minimum(c1["visibility_score"][i].astype(np.float32),c2["visibility_score"][j].astype(np.float32))*joint.astype(np.float32)
    else:conf=joint.astype(np.float32)
    masked=np.where(joint,ps,0.0).astype(np.float32)
    den=float(conf.sum())
    weighted=float((ps*conf).sum()/max(den,1e-6)) if den>0 else 0.0
    vals=ps[joint]
    pmax=float(vals.max()) if len(vals) else 0.0
    pstd=float(vals.std()) if len(vals)>1 else 0.0
    return masked,conf,float(joint.mean()),weighted,pmax,pstd


def pair_feature_np(c, i:int, j:int, *, neighbor_jaccard: float = 0.0) -> np.ndarray:
    gf=float(np.dot(c["z_fused"][i],c["z_fused"][j]));gg=float(np.dot(c["z_global"][i],c["z_global"][j]))
    glocal=float(np.dot(c["z_local"][i],c["z_local"][j])) if "z_local" in c else 0.0
    ql=c["local"][i].astype(np.float32);gl=c["local"][j].astype(np.float32);sim=ql@gl.T
    local=float(.5*(sim.max(1).mean()+sim.max(0).mean()))
    masked,conf,vis_count,pmean,pmax,pstd=_part_evidence(c,i,c,j)
    cd=float(np.linalg.norm(c["color"][i]-c["color"][j]));color_sim=float(np.exp(-3.0*cd))
    same_cam=float(str(c["camera_id"][i])==str(c["camera_id"][j])) if "camera_id" in c else 0.0
    base=[gf,gg,glocal,local,color_sim,vis_count,pmean,pmax,pstd,same_cam,float(neighbor_jaccard)]
    return np.concatenate([np.asarray(base,np.float32),masked,conf]).astype(np.float32)


def pair_feature_np_cross(q,i:int,g,j:int,*,neighbor_jaccard:float=0.0)->np.ndarray:
    gf=float(np.dot(q["z_fused"][i],g["z_fused"][j]));gg=float(np.dot(q["z_global"][i],g["z_global"][j]))
    glocal=float(np.dot(q["z_local"][i],g["z_local"][j])) if "z_local" in q and "z_local" in g else 0.0
    ql=q["local"][i].astype(np.float32);gl=g["local"][j].astype(np.float32);sim=ql@gl.T
    local=float(.5*(sim.max(1).mean()+sim.max(0).mean()))
    masked,conf,vis_count,pmean,pmax,pstd=_part_evidence(q,i,g,j)
    cd=float(np.linalg.norm(q["color"][i]-g["color"][j]));color_sim=float(np.exp(-3.0*cd))
    same_cam=float(str(q["camera_id"][i])==str(g["camera_id"][j])) if "camera_id" in q and "camera_id" in g else 0.0
    base=[gf,gg,glocal,local,color_sim,vis_count,pmean,pmax,pstd,same_cam,float(neighbor_jaccard)]
    return np.concatenate([np.asarray(base,np.float32),masked,conf]).astype(np.float32)


def pair_features_np_cross_batch(q, i: int, g, js, *, neighbor_jaccard=None) -> np.ndarray:
    """Vectorized equivalent of :func:`pair_feature_np_cross` for one query vs K gallery items.

    Retrieval calls this on the rerank head (typically top-100).  The old Python loop built every
    pair separately and dominated full-pipeline latency.  Keep the exact feature layout so existing
    reranker weights remain compatible.
    """
    js=np.asarray(js,dtype=np.int64).reshape(-1); K=len(js)
    if K==0:
        slots=int(np.asarray(q["parts"]).shape[1])
        return np.zeros((0,11+2*slots),np.float32)
    zf_q=np.asarray(q["z_fused"][i],np.float32); zf_g=np.asarray(g["z_fused"][js],np.float32)
    zg_q=np.asarray(q["z_global"][i],np.float32); zg_g=np.asarray(g["z_global"][js],np.float32)
    gf=zf_g@zf_q; gg=zg_g@zg_q
    if "z_local" in q and "z_local" in g:
        zl_q=np.asarray(q["z_local"][i],np.float32); zl_g=np.asarray(g["z_local"][js],np.float32); glocal=zl_g@zl_q
    else: glocal=np.zeros(K,np.float32)

    ql=np.asarray(q["local"][i],np.float32); gl=np.asarray(g["local"][js],np.float32)
    sim=np.einsum("ld,kmd->klm",ql,gl,optimize=True)
    local=.5*(sim.max(axis=2).mean(axis=1)+sim.max(axis=1).mean(axis=1))

    qp=np.asarray(q["parts"][i],np.float32); gp=np.asarray(g["parts"][js],np.float32)
    ps=np.einsum("sd,ksd->ks",qp,gp,optimize=True).astype(np.float32,copy=False)
    qv=np.asarray(q["visibility"][i],bool); gv=np.asarray(g["visibility"][js],bool); joint=gv & qv[None,:]
    if "visibility_score" in q and "visibility_score" in g:
        qvs=np.asarray(q["visibility_score"][i],np.float32); gvs=np.asarray(g["visibility_score"][js],np.float32)
        conf=np.minimum(gvs,qvs[None,:])*joint.astype(np.float32)
    else: conf=joint.astype(np.float32)
    masked=np.where(joint,ps,0.0).astype(np.float32)
    den=conf.sum(axis=1); pmean=np.divide((ps*conf).sum(axis=1),np.maximum(den,1e-6),out=np.zeros(K,np.float32),where=den>0)
    cnt=joint.sum(axis=1).astype(np.float32); vis_count=joint.mean(axis=1,dtype=np.float32)
    pmax=np.where(cnt>0,np.where(joint,ps,-np.inf).max(axis=1),0.0).astype(np.float32)
    raw_mean=np.divide((ps*joint).sum(axis=1),np.maximum(cnt,1.0),out=np.zeros(K,np.float32),where=cnt>0)
    raw_m2=np.divide(((ps*ps)*joint).sum(axis=1),np.maximum(cnt,1.0),out=np.zeros(K,np.float32),where=cnt>0)
    pstd=np.sqrt(np.maximum(raw_m2-raw_mean*raw_mean,0.0)).astype(np.float32)
    pstd=np.where(cnt>1,pstd,0.0).astype(np.float32)

    qc=np.asarray(q["color"][i],np.float32); gc=np.asarray(g["color"][js],np.float32)
    color_sim=np.exp(-3.0*np.linalg.norm(gc-qc[None,:],axis=1)).astype(np.float32)
    if "camera_id" in q and "camera_id" in g:
        same_cam=(np.asarray(g["camera_id"])[js].astype(str)==str(q["camera_id"][i])).astype(np.float32)
    else: same_cam=np.zeros(K,np.float32)
    if neighbor_jaccard is None: jac=np.zeros(K,np.float32)
    else:
        jac=np.asarray(neighbor_jaccard,np.float32).reshape(-1)
        if len(jac)!=K: raise ValueError("neighbor_jaccard length mismatch")
    base=np.stack([gf,gg,glocal,local,color_sim,vis_count,pmean,pmax,pstd,same_cam,jac],axis=1).astype(np.float32)
    return np.concatenate([base,masked,conf.astype(np.float32)],axis=1).astype(np.float32,copy=False)


def build_pair_training(cache_path: str|Path, out_npz: str|Path, hard_map_path: str|Path|None=None,
                        positives_per_id=6, negatives_per_id=12, seed=42, knn_k=20):
    """Legacy pairwise dataset retained for ablations. New default is listwise."""
    c=load_cache(cache_path);keys=c["vehicle_key"].astype(str);cams=c["camera_id"].astype(str)
    by={k:np.flatnonzero(keys==k).tolist() for k in sorted(set(keys))};hard=json.load(open(hard_map_path,encoding="utf-8")) if hard_map_path else {}
    knn=build_same_cache_knn(c,knn_k);rng=random.Random(seed);X=[];y=[];anchor_ids=[];all_keys=list(by)
    for k,inds in by.items():
        pos=[]
        for a in inds:
            opts=[b for b in inds if b!=a and cams[b]!=cams[a]];rng.shuffle(opts)
            if opts:pos.append((a,opts[0]))
        rng.shuffle(pos)
        for a,b in pos[:positives_per_id]:X.append(pair_feature_np(c,a,b,neighbor_jaccard=_jaccard(knn[a],knn[b])));y.append(1);anchor_ids.append(k)
        neg_keys=[x for x in hard.get(k,[]) if x in by and x!=k] or [x for x in all_keys if x!=k]
        for _ in range(negatives_per_id):
            nk=rng.choice(neg_keys);a=rng.choice(inds);b=rng.choice(by[nk]);X.append(pair_feature_np(c,a,b,neighbor_jaccard=_jaccard(knn[a],knn[b])));y.append(0);anchor_ids.append(k)
    X=np.stack(X);y=np.asarray(y,dtype=np.float32);anchor_ids=np.asarray(anchor_ids,dtype=np.str_);p=Path(out_npz);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,X=X,y=y,anchor_vehicle_key=anchor_ids);return X.shape


def build_listwise_training(cache_path: str|Path, out_npz: str|Path, hard_map_path: str|Path|None=None,
                            candidates_per_query: int = 24, groups_per_id: int = 4, seed: int = 42, knn_k: int = 20):
    """Build candidate sets that directly model top-of-ranking competition behaviour."""
    c=load_cache(cache_path);keys=c["vehicle_key"].astype(str);cams=c["camera_id"].astype(str);by={k:np.flatnonzero(keys==k).tolist() for k in sorted(set(keys))}
    hard=json.load(open(hard_map_path,encoding="utf-8")) if hard_map_path else {};knn=build_same_cache_knn(c,knn_k);rng=random.Random(seed);all_keys=list(by);Xs=[];targets=[];anchors=[]
    C=max(2,int(candidates_per_query))
    for k,inds in by.items():
        anchors_with_pos=[]
        for a in inds:
            pos=[b for b in inds if b!=a and cams[b]!=cams[a]]
            if pos: anchors_with_pos.append((a,pos))
        rng.shuffle(anchors_with_pos)
        for a,pos in anchors_with_pos[:max(1,int(groups_per_id))]:
            pidx=rng.choice(pos);neg_keys=[x for x in hard.get(k,[]) if x in by and x!=k]
            if len(neg_keys)<C-1:
                extras=[x for x in all_keys if x!=k and x not in neg_keys];rng.shuffle(extras);neg_keys+=extras
            neg=[]
            for nk in neg_keys:
                if len(neg)>=C-1:break
                neg.append(rng.choice(by[nk]))
            while len(neg)<C-1:
                nk=rng.choice([x for x in all_keys if x!=k]);neg.append(rng.choice(by[nk]))
            cand=[pidx]+neg; rng.shuffle(cand); target=cand.index(pidx)
            X=np.stack([pair_feature_np(c,a,j,neighbor_jaccard=_jaccard(knn[a],knn[j])) for j in cand])
            Xs.append(X);targets.append(target);anchors.append(k)
    X=np.stack(Xs).astype(np.float32);t=np.asarray(targets,np.int64);anchors=np.asarray(anchors,dtype=np.str_);p=Path(out_npz);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,X=X,target=t,anchor_vehicle_key=anchors)
    return X.shape


def _split_anchor_ids(anchors, seed=42, frac=.10):
    uniq=np.array(sorted(set(anchors.tolist())),dtype=str);rng=np.random.default_rng(seed);rng.shuffle(uniq);n=max(1,int(round(frac*len(uniq))));va_ids=set(uniq[:n].tolist());va=np.flatnonzero(np.isin(anchors,list(va_ids)));tr=np.flatnonzero(~np.isin(anchors,list(va_ids)));return tr,va


def train_reranker(pair_npz: str|Path, out_pt: str|Path, epochs=20, batch=2048, lr=2e-3, device="cuda", seed=42, mode="auto"):
    np.random.seed(seed);torch.manual_seed(seed)
    with np.load(pair_npz,allow_pickle=False) as d:
        anchors=d["anchor_vehicle_key"].astype(str) if "anchor_vehicle_key" in d.files else None
        is_list="target" in d.files and d["X"].ndim==3
        X=d["X"].astype(np.float32); target=d["target"].astype(np.int64) if is_list else d["y"].astype(np.float32)
    if mode=="listwise" and not is_list:raise ValueError("Requested listwise mode but pair file is pairwise")
    if anchors is not None:tr,va=_split_anchor_ids(anchors,seed)
    else:
        perm=np.random.permutation(len(target));cut=int(len(target)*.9);tr,va=perm[:cut],perm[cut:]
    dev=torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu");model=PairReranker(X.shape[-1]).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3);best=1e9;best_state=None
    if is_list:
        ds=TensorDataset(torch.from_numpy(X[tr]),torch.from_numpy(target[tr]));dl=DataLoader(ds,batch_size=max(8,min(batch,256)),shuffle=True)
        for ep in range(1,epochs+1):
            model.train();losses=[]
            for xb,yb in dl:
                xb=xb.to(dev);yb=yb.to(dev);B,C,D=xb.shape;s=model(xb.reshape(B*C,D)).reshape(B,C);ce=torch.nn.functional.cross_entropy(s,yb)
                pos=s.gather(1,yb[:,None]).squeeze(1);mask=torch.nn.functional.one_hot(yb,C).bool();hard=s.masked_fill(mask,-1e9).max(1).values;rank=torch.nn.functional.softplus(hard-pos+0.10).mean();loss=ce+0.25*rank
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():
                xv=torch.from_numpy(X[va]).to(dev);yv=torch.from_numpy(target[va]).to(dev);B,C,D=xv.shape;sv=model(xv.reshape(B*C,D)).reshape(B,C);vl=float(torch.nn.functional.cross_entropy(sv,yv).cpu()) if len(va) else float(np.mean(losses));acc=float((sv.argmax(1)==yv).float().mean().cpu()) if len(va) else 0.
            print(f"reranker(listwise) epoch={ep} train={np.mean(losses):.5f} val_ce={vl:.5f} val_top1={acc:.4f} val_anchor_ids={len(set(anchors[va])) if anchors is not None else 'legacy'}")
            if vl<best:best=vl;best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        metadata={"mode":"listwise","candidates_per_query":int(X.shape[1])}
    else:
        y=target;pos=float(y[tr].sum());neg=float(len(tr)-pos);pos_weight=torch.tensor([neg/max(1.,pos)],device=dev);ds=TensorDataset(torch.from_numpy(X[tr]),torch.from_numpy(y[tr]));dl=DataLoader(ds,batch_size=batch,shuffle=True)
        for ep in range(1,epochs+1):
            model.train();losses=[]
            for xb,yb in dl:
                xb=xb.to(dev);yb=yb.to(dev);logit=model(xb);loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,yb,pos_weight=pos_weight);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():xv=torch.from_numpy(X[va]).to(dev);yv=torch.from_numpy(y[va]).to(dev);vl=float(torch.nn.functional.binary_cross_entropy_with_logits(model(xv),yv).cpu()) if len(va) else float(np.mean(losses))
            print(f"reranker(pairwise) epoch={ep} train={np.mean(losses):.5f} val={vl:.5f}")
            if vl<best:best=vl;best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        metadata={"mode":"pairwise"}
    Path(out_pt).parent.mkdir(parents=True,exist_ok=True);torch.save({"model":best_state,"input_dim":X.shape[-1],**metadata},out_pt);return out_pt


def load_reranker(path: str|Path, device="cpu"):
    ck=torch.load(path,map_location=device,weights_only=False);m=PairReranker(int(ck["input_dim"]));m.load_state_dict(ck["model"]);return m.to(device).eval()


def concat_pair_datasets(paths, out_npz):
    """Concatenate scalar pair/listwise features from different OOF backbone folds safely.

    Raw embeddings from independently trained folds are not in the same coordinate system, but
    derived cosine/local/part features are.  Therefore cross-fold reranker training must concatenate
    *pair features*, never raw OOF embeddings.
    """
    payloads=[]
    for p in paths:
        with np.load(p,allow_pickle=False) as d: payloads.append({k:d[k] for k in d.files})
    if not payloads: raise ValueError('No pair datasets')
    is_list='target' in payloads[0]
    if any(('target' in x)!=is_list for x in payloads): raise ValueError('Cannot mix pairwise and listwise datasets')
    X=np.concatenate([x['X'] for x in payloads],axis=0); anchors=np.concatenate([x['anchor_vehicle_key'] for x in payloads],axis=0)
    out={'X':X,'anchor_vehicle_key':anchors}
    if is_list: out['target']=np.concatenate([x['target'] for x in payloads],axis=0)
    else: out['y']=np.concatenate([x['y'] for x in payloads],axis=0)
    p=Path(out_npz);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**out);return p
