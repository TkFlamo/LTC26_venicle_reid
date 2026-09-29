from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from .features import load_cache
from .official_eval import write_candidates, write_embeddings, write_submission
from .official_validation import rank_caches
from .refusal import refusal_probability
from .pairs import pair_feature_np_cross  # backward-compatible public import


def _normalize(x):
    x=x.astype(np.float32,copy=True);x/=np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12);return x


def _cache_ids(cache, requested: str | None):
    col=requested or ("image_id" if "meta_image_id" in cache else "sample_id")
    if col=="sample_id": return cache["sample_id"].astype(str)
    key=f"meta_{col}"
    if key not in cache: raise KeyError(f"{col} was not stored in cache; available metadata: {[k for k in cache if k.startswith('meta_')]}")
    return cache[key].astype(str)


def run_retrieval(
    query_cache: str|Path, gallery_cache: str|Path, out_dir: str|Path, *,
    reranker_path: str|Path|None=None, refusal_json: str|Path|None=None,
    retrieval_recipe_json: str|Path|None=None, ann_topk: int=100, output_topk: int=10,
    query_id_column: str|None=None, gallery_id_column: str|None=None,
):
    q=load_cache(query_cache);g=load_cache(gallery_cache)
    refusal=json.load(open(refusal_json,encoding="utf-8")) if refusal_json else None
    if retrieval_recipe_json: recipe=json.load(open(retrieval_recipe_json,encoding="utf-8"))
    elif refusal and refusal.get("retrieval_recipe"): recipe=refusal["retrieval_recipe"]
    else: recipe={"base_alpha":0.0,"reranker_beta":0.0,"rerank_topk":ann_topk,"kreciprocal_lambda":0.0,"kreciprocal_k":20,"same_camera_filter":False}
    ranked,ranked_scores,qe,ge,_,_=rank_caches(
        q,g,base_alpha=float(recipe.get("base_alpha",0.0)),reranker_path=reranker_path,
        reranker_beta=float(recipe.get("reranker_beta",0.0)),rerank_topk=int(recipe.get("rerank_topk",ann_topk)),
        kreciprocal_lambda=float(recipe.get("kreciprocal_lambda",0.0)),kreciprocal_k=int(recipe.get("kreciprocal_k",20)),
        same_camera_filter=bool(recipe.get("same_camera_filter",False)),device="cpu",
    )
    qids=_cache_ids(q,query_id_column);gids=_cache_ids(g,gallery_id_column);gid_to_idx={str(x):i for i,x in enumerate(gids.tolist())}
    # rank_caches uses cache ids; remap only if caller explicitly requests alternate IDs.
    cache_qids=q["meta_image_id"].astype(str) if "meta_image_id" in q else q["sample_id"].astype(str)
    cache_gids=g["meta_image_id"].astype(str) if "meta_image_id" in g else g["sample_id"].astype(str)
    qmap={str(a):str(b) for a,b in zip(cache_qids,qids)};gmap={str(a):str(b) for a,b in zip(cache_gids,gids)}
    ranked_out={qmap.get(k,k):[gmap.get(x,x) for x in v] for k,v in ranked.items()}
    candidate_rows=[];qglob=_normalize(q["z_global"]);gglob=_normalize(g["z_global"]);cache_gid_index={str(x):i for i,x in enumerate(cache_gids)}
    for qi,cqid in enumerate(cache_qids):
        order=ranked.get(str(cqid),[]);scores=ranked_scores.get(str(cqid),np.zeros(0,np.float32));accepted=bool(len(order));match_prob=float(scores[0]) if len(scores) else 0.0
        if refusal and len(order):
            top_gid=str(order[0]);gi=cache_gid_index[top_gid];s1=float(scores[0]);s2=float(scores[1]) if len(scores)>1 else 0.;top5=float(np.mean(scores[:min(5,len(scores))]));glob=float(qglob[qi]@gglob[gi]);X=np.asarray([[s1,s1-s2,glob,s1-top5]],np.float32);match_prob=float(refusal_probability(refusal,X)[0]);accepted=match_prob>=float(refusal["threshold"])
        elif refusal: accepted=False
        if accepted and order: candidate_rows.append((qmap.get(str(cqid),str(cqid)),gmap.get(str(order[0]),str(order[0])),float(match_prob)))
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);write_submission(out/"submission.csv",ranked_out,[str(x) for x in qids],top_k=output_topk);write_candidates(out/"candidates.csv",candidate_rows);write_embeddings(out/"embeddings.npy",qe,ge);(out/"retrieval_recipe_used.json").write_text(json.dumps(recipe,ensure_ascii=False,indent=2),encoding="utf-8");return out
