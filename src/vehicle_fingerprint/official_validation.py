from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from .features import load_cache
from .official_eval import (
    blended_embedding, load_official_evaluator, normalize, official_selection_key,
    write_candidates, write_embeddings, write_submission,
)
from .pairs import load_reranker, pair_feature_np_cross, pair_features_np_cross_batch, build_cross_knn_context


def _cache_ids(c: dict) -> np.ndarray:
    if "meta_image_id" in c:
        return c["meta_image_id"].astype(str)
    return c["sample_id"].astype(str)


def _reranker_scores_cross(q, qi: int, g, cand: np.ndarray, reranker, device: str, qknn=None, gknn=None, batch_size: int = 2048) -> np.ndarray:
    if len(cand)==0: return np.zeros(0,np.float32)
    if qknn is not None and gknn is not None:
        a=qknn[qi]; jac=[]
        for gi in cand:
            b=gknn[int(gi)]; u=len(a|b); jac.append(float(len(a&b)/u) if u else 0.0)
    else: jac=None
    feats=pair_features_np_cross_batch(q,qi,g,cand,neighbor_jaccard=jac)
    dev=torch.device(device); out=[]
    with torch.inference_mode():
        for st in range(0,len(feats),batch_size):
            xb=torch.from_numpy(feats[st:st+batch_size]).to(dev)
            out.append(torch.sigmoid(reranker(xb)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def rank_caches(
    query_cache: str|Path|dict,
    gallery_cache: str|Path|dict,
    *,
    base_alpha: float=0.0,
    reranker_path: str|Path|None=None,
    reranker_beta: float=0.0,
    rerank_topk: int=100,
    kreciprocal_lambda: float=0.0,
    kreciprocal_k: int=20,
    same_camera_filter: bool=False,
    device: str="cpu",
):
    q=load_cache(query_cache) if not isinstance(query_cache,dict) else query_cache
    g=load_cache(gallery_cache) if not isinstance(gallery_cache,dict) else gallery_cache
    qe=blended_embedding(q["z_global"],q["z_fused"],base_alpha); ge=blended_embedding(g["z_global"],g["z_fused"],base_alpha)
    sim=normalize(qe)@normalize(ge).T; qids=_cache_ids(q); gids=_cache_ids(g)
    beta=float(np.clip(reranker_beta,0,1)); kl=float(np.clip(kreciprocal_lambda,0,1))
    rr=load_reranker(reranker_path,device) if reranker_path and beta>0 else None
    need_knn = kl>0 or rr is not None
    qknn,gknn=build_cross_knn_context(q,g,kreciprocal_k) if need_knn else (None,None)
    ranked={}; ranked_scores={}
    qcams=q.get("camera_id"); gcams=g.get("camera_id")
    camera_filter_available=False
    if same_camera_filter and qcams is not None and gcams is not None:
        qall={str(x) for x in np.asarray(qcams).tolist()}; gall={str(x) for x in np.asarray(gcams).tolist()}
        missing_tokens={"-1","","none","nan","unknown"}
        camera_filter_available=bool((qall-missing_tokens) and (gall-missing_tokens))
    for qi,qid in enumerate(qids.tolist()):
        base_order=np.argsort(-sim[qi],kind="stable")
        if camera_filter_available:
            qc=str(qcams[qi]); base_order=np.asarray([gi for gi in base_order if str(gcams[int(gi)])!=qc],dtype=np.int64)
        order=base_order; base01=np.clip((sim[qi,base_order]+1.0)*0.5,0,1).astype(np.float32)
        pre=base01.copy()
        if kl>0 and len(base_order):
            jac=np.asarray([len(qknn[qi]&gknn[int(gi)])/max(1,len(qknn[qi]|gknn[int(gi)])) for gi in base_order],np.float32)
            pre=(1.0-kl)*pre+kl*jac
            ro=np.argsort(-pre,kind="stable");order=base_order[ro];pre=pre[ro]
        scores=pre
        if rr is not None and len(order):
            k=min(int(rerank_topk),len(order));head=order[:k]
            rrs=_reranker_scores_cross(q,qi,g,head,rr,device,qknn=qknn,gknn=gknn)
            head_base=np.asarray([scores[np.flatnonzero(order==gi)[0]] for gi in head],np.float32)
            comb=(1.0-beta)*head_base+beta*rrs;ro=np.argsort(-comb,kind="stable");head2=head[ro];score2=comb[ro]
            tail=order[k:];tail_score=scores[k:];order=np.concatenate([head2,tail]);scores=np.concatenate([score2,tail_score])
        ranked[str(qid)]=[str(gids[i]) for i in order];ranked_scores[str(qid)]=scores
    return ranked,ranked_scores,qe,ge,q,g


def evaluate_recipe(
    query_cache, gallery_cache, gt_csv, *, base_alpha=0.0, reranker_path=None,
    reranker_beta=0.0, rerank_topk=100, kreciprocal_lambda=0.0, kreciprocal_k=20,
    same_camera_filter=False, device="cpu", evaluator_path=None, top_k=10,
):
    ranked,scores,qe,ge,q,g=rank_caches(query_cache,gallery_cache,base_alpha=base_alpha,reranker_path=reranker_path,reranker_beta=reranker_beta,rerank_topk=rerank_topk,kreciprocal_lambda=kreciprocal_lambda,kreciprocal_k=kreciprocal_k,same_camera_filter=same_camera_filter,device=device)
    mod=load_official_evaluator(evaluator_path); query,gallery=mod.load_gt(str(gt_csv))
    submission_ranked={qid:vals[:int(top_k)] for qid,vals in ranked.items()}
    report={"ranking":mod.ranking_metrics(query,gallery,submission_ranked,top_k=int(top_k),ranks=(1,5))}
    qids=_cache_ids(q).tolist(); gids=_cache_ids(g).tolist()
    report["full_ranking"]=mod.full_ranking_metrics(normalize(qe),normalize(ge),qids,gids,query,gallery)
    return report,ranked,scores,qe,ge


def select_retrieval_recipe_official(
    val_query_cache, val_gallery_cache, gt_csv, *, reranker_path=None, rerank_topk=100,
    alpha_grid=None, beta_grid=None, kreciprocal_lambda_grid=None, same_camera_filter_grid=None,
    kreciprocal_k=20, device="cpu", evaluator_path=None,
):
    alpha_grid=alpha_grid or [0,.03,.05,.1,.15,.25,.4,.7,1.0]
    beta_grid=beta_grid or [0,.1,.2,.35,.5]
    kreciprocal_lambda_grid=kreciprocal_lambda_grid or [0.0]
    same_camera_filter_grid=same_camera_filter_grid or [False]
    rows=[];best=None
    for alpha in alpha_grid:
      for beta in beta_grid:
       if beta>0 and not reranker_path: continue
       for kl in kreciprocal_lambda_grid:
        for scf in same_camera_filter_grid:
            report,_,_,_,_=evaluate_recipe(val_query_cache,val_gallery_cache,gt_csv,base_alpha=float(alpha),reranker_path=reranker_path,reranker_beta=float(beta),rerank_topk=rerank_topk,kreciprocal_lambda=float(kl),kreciprocal_k=int(kreciprocal_k),same_camera_filter=bool(scf),device=device,evaluator_path=evaluator_path)
            key=official_selection_key(report);r=report["ranking"];f=report["full_ranking"]
            row={"base_alpha":float(alpha),"reranker_beta":float(beta),"kreciprocal_lambda":float(kl),"same_camera_filter":bool(scf),"mAP@10":r["mAP@10"],"Rank-1":r["Rank-1"],"Rank-5":r["Rank-5"],"mAP_full":f["mAP_full"],"mINP":f["mINP"]};rows.append(row)
            if best is None or key>best[0]:best=(key,row)
    if best is None:raise RuntimeError("No retrieval recipe candidates were evaluated")
    recipe={"base_alpha":best[1]["base_alpha"],"reranker_beta":best[1]["reranker_beta"],"rerank_topk":int(rerank_topk),"kreciprocal_lambda":best[1]["kreciprocal_lambda"],"kreciprocal_k":int(kreciprocal_k),"same_camera_filter":best[1]["same_camera_filter"],"selection":"official_mAP@10_then_Rank1_Rank5_full_mAP_mINP"}
    return recipe,pd.DataFrame(rows).sort_values(["mAP@10","Rank-1","Rank-5"],ascending=False,kind="stable")


def _refusal_features(query_cache, gallery_cache, ranked, ranked_scores, gt_csv, evaluator_path=None):
    q=load_cache(query_cache); g=load_cache(gallery_cache); qids=_cache_ids(q); gids=_cache_ids(g)
    gid_to_index={str(x):i for i,x in enumerate(gids.tolist())}
    qid_to_index={str(x):i for i,x in enumerate(qids.tolist())}
    qg=normalize(q["z_global"]); gg=normalize(g["z_global"])
    mod=load_official_evaluator(evaluator_path); query,gallery=mod.load_gt(str(gt_csv))
    X=[]; y=[]; meta=[]
    for qid,row in query.iterrows():
        qid=str(qid)
        if qid not in qid_to_index: continue
        order=ranked.get(qid,[]); scores=ranked_scores.get(qid,np.zeros(0,np.float32))
        if not order or len(scores)==0: continue
        qi=qid_to_index[qid]; top_gid=str(order[0]); gi=gid_to_index.get(top_gid)
        if gi is None: continue
        s1=float(scores[0]); s2=float(scores[1]) if len(scores)>1 else 0.0; top5=float(np.mean(scores[:min(5,len(scores))]))
        glob=float(qg[qi]@gg[gi])
        X.append([s1,s1-s2,glob,s1-top5]); y.append(1 if mod.valid_positives(row,gallery)>0 else 0)
        meta.append((qid,top_gid))
    return np.asarray(X,np.float32),np.asarray(y,np.int64),meta,query,gallery


def fit_refusal_official(
    val_query_cache, val_gallery_cache, gt_csv, retrieval_recipe, out_json, *, reranker_path=None,
    device="cpu", evaluator_path=None, seed=42,
):
    report,ranked,scores,_,_=evaluate_recipe(
        val_query_cache,val_gallery_cache,gt_csv,base_alpha=retrieval_recipe["base_alpha"],reranker_path=reranker_path,
        reranker_beta=retrieval_recipe["reranker_beta"],rerank_topk=retrieval_recipe["rerank_topk"],kreciprocal_lambda=retrieval_recipe.get("kreciprocal_lambda",0.0),kreciprocal_k=retrieval_recipe.get("kreciprocal_k",20),same_camera_filter=retrieval_recipe.get("same_camera_filter",False),device=device,evaluator_path=evaluator_path,
    )
    X,y,meta,query,gallery=_refusal_features(val_query_cache,val_gallery_cache,ranked,scores,gt_csv,evaluator_path)
    if len(np.unique(y))<2:
        raise RuntimeError("Official validation protocol needs both known and open-set queries for refusal calibration")
    clf=LogisticRegression(class_weight="balanced",max_iter=2000,random_state=int(seed)).fit(X,y)
    probs=clf.predict_proba(X)[:,1]
    mod=load_official_evaluator(evaluator_path); best=None
    for t in np.linspace(.01,.99,197):
        cand={}
        for (qid,gid),p in zip(meta,probs):
            if p>=t: cand[str(qid)]=[(str(gid),float(p))]
        m=mod.candidate_metrics(query,gallery,cand)
        key=(float(m["F1"]),float(m["TNR"]) if np.isfinite(m["TNR"]) else -1.0,float(m["PR-AUC"]) if np.isfinite(m["PR-AUC"]) else -1.0)
        if best is None or key>best[0]: best=(key,float(t),m)
    spec={
        "coef":clf.coef_[0].tolist(),"intercept":float(clf.intercept_[0]),"threshold":best[1],
        "features":["top1","margin12","global_top1","top1_minus_top5mean"],
        "calibration_metrics":best[2],"retrieval_recipe":retrieval_recipe,
        "selection":"official_candidates_F1_then_TNR_then_PR-AUC","seed":int(seed),
    }
    out=Path(out_json);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding="utf-8")
    return spec


def _refusal_probability(spec,X):
    w=np.asarray(spec["coef"],np.float32); b=float(spec["intercept"]); z=np.asarray(X,np.float32)@w+b
    return 1/(1+np.exp(-z))


def generate_official_artifacts(
    query_cache, gallery_cache, gt_csv, out_dir, retrieval_recipe, refusal_spec, *, reranker_path=None,
    device="cpu", evaluator_path=None, top_k=10,
):
    report,ranked,scores,qe,ge=evaluate_recipe(
        query_cache,gallery_cache,gt_csv,base_alpha=retrieval_recipe["base_alpha"],reranker_path=reranker_path,
        reranker_beta=retrieval_recipe["reranker_beta"],rerank_topk=retrieval_recipe["rerank_topk"],kreciprocal_lambda=retrieval_recipe.get("kreciprocal_lambda",0.0),kreciprocal_k=retrieval_recipe.get("kreciprocal_k",20),same_camera_filter=retrieval_recipe.get("same_camera_filter",False),device=device,evaluator_path=evaluator_path,top_k=top_k,
    )
    X,y,meta,query,gallery=_refusal_features(query_cache,gallery_cache,ranked,scores,gt_csv,evaluator_path)
    probs=_refusal_probability(refusal_spec,X)
    threshold=float(refusal_spec["threshold"]); cand_rows=[]
    for (qid,gid),p in zip(meta,probs):
        if p>=threshold: cand_rows.append((str(qid),str(gid),float(p)))
    q=load_cache(query_cache); qids=_cache_ids(q).tolist()
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    write_submission(out/"submission.csv",ranked,qids,top_k=top_k)
    write_candidates(out/"candidates.csv",cand_rows)
    write_embeddings(out/"embeddings.npy",qe,ge)
    return out


def run_official_script(
    gt_csv, submission, *, candidates=None, embeddings=None, query_csv=None, gallery_csv=None,
    json_out=None, evaluator_path=None,
):
    evaluator=Path(evaluator_path) if evaluator_path else Path(__file__).resolve().parents[2]/"official"/"evaluate.py"
    cmd=[sys.executable,str(evaluator),"--gt",str(gt_csv),"--submission",str(submission)]
    if candidates: cmd += ["--candidates",str(candidates)]
    if embeddings:
        if not query_csv or not gallery_csv: raise ValueError("embeddings require query_csv and gallery_csv")
        cmd += ["--embeddings",str(embeddings),"--query",str(query_csv),"--gallery",str(gallery_csv)]
    if json_out: cmd += ["--json",str(json_out)]
    subprocess.run(cmd,check=True)
    return json.loads(Path(json_out).read_text(encoding="utf-8")) if json_out else None
