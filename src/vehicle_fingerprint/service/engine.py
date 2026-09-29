from __future__ import annotations

import json
import numpy as np
import torch
from PIL import Image

from ..data.crop import crop_bbox
from ..data.color import robust_color_descriptor
from ..data.dataset import _baseline_v4_transform
from ..features import load_cache, load_inference_model
from ..pairs import load_reranker
from ..refusal import refusal_probability
from ..retrieval import pair_feature_np_cross
from ..utils import autocast_context


def _norm(x):
    x=x.astype(np.float32,copy=True);x/=np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12);return x

def _recipe_embedding(cache,alpha):
    a=float(np.clip(alpha,0,1));zg=_norm(cache['z_global']);zf=_norm(cache['z_fused'])
    if a<=0:return zg
    if a>=1:return zf
    return np.concatenate([np.sqrt(1-a)*zg,np.sqrt(a)*zf],1).astype(np.float32)

class VehicleSearchEngine:
    def __init__(self,checkpoint,gallery_cache,*,reranker=None,refusal=None,retrieval_recipe=None,device='0',precision='bf16',image_size=None,**_unused):
        self.model,self.device,mcfg=load_inference_model(checkpoint,device);self.gallery=load_cache(gallery_cache);self.refusal=json.load(open(refusal,encoding='utf-8')) if refusal else None
        if retrieval_recipe:self.recipe=json.load(open(retrieval_recipe,encoding='utf-8'))
        elif self.refusal and self.refusal.get('retrieval_recipe'):self.recipe=self.refusal['retrieval_recipe']
        else:self.recipe={'base_alpha':0.,'reranker_beta':0.,'rerank_topk':100}
        self.alpha=float(np.clip(self.recipe.get('base_alpha',0),0,1));self.beta=float(np.clip(self.recipe.get('reranker_beta',0),0,1));self.reranker=load_reranker(reranker,'cpu') if reranker and self.beta>0 else None;self.precision=precision
        prep=mcfg.get('preprocess',{});self.image_size=image_size or prep.get('image_size',[256,384]);self.transform=_baseline_v4_transform(self.image_size,False);self.gallery_z=_recipe_embedding(self.gallery,self.alpha)
        try:
            import faiss;self.index=faiss.IndexFlatIP(self.gallery_z.shape[1]);self.index.add(self.gallery_z)
        except Exception:self.index=None
    def _query_features(self,image):
        x=self.transform(image.convert('RGB'))
        with torch.inference_mode():
            with autocast_context(self.device,self.precision):o=self.model(x[None].to(self.device))
        q={k:o[k].float().cpu().numpy() for k in ('z_fused','z_global','z_local','parts','local')};q['visibility']=o['visibility'].cpu().numpy().astype(np.uint8);q['color']=robust_color_descriptor(image.convert('RGB'),None)[None];q['sample_id']=np.array(['query']);q['vehicle_key']=np.array(['query']);q['camera_id']=np.array(['-1']);return q
    def search(self,image,bbox=None,topk=10,ann_topk=100):
        if bbox is not None:image=crop_bbox(image,*bbox,pad=.03)
        image=image.convert('RGB');q=self._query_features(image);z=_recipe_embedding(q,self.alpha);k=min(int(self.recipe.get('rerank_topk',ann_topk)),len(self.gallery_z))
        if self.index is not None:D,I=self.index.search(z,k);inds,ann=I[0],D[0]
        else:s=(z@self.gallery_z.T)[0];inds=np.argsort(-s)[:k];ann=s[inds]
        pairs=[]
        for rank,gi in enumerate(inds):
            gi=int(gi);base=float(ann[rank]);base01=float(np.clip((base+1)*.5,0,1))
            if self.reranker:
                feat=pair_feature_np_cross(q,0,self.gallery,gi)
                with torch.inference_mode():rr=float(torch.sigmoid(self.reranker(torch.from_numpy(feat)[None])).item())
                score=(1-self.beta)*base01+self.beta*rr
            else:score=base01
            pairs.append((gi,score,base))
        pairs.sort(key=lambda x:x[1],reverse=True);pairs=pairs[:topk];scores=[p[1] for p in pairs];accepted=True;prob=scores[0] if scores else 0.
        if self.refusal and scores:
            s1=scores[0];s2=scores[1] if len(scores)>1 else 0.;top5=float(np.mean(scores[:min(5,len(scores))]));top_gi=pairs[0][0];glob=float(np.dot(_norm(q['z_global'])[0],_norm(self.gallery['z_global'][[top_gi]])[0]));prob=float(refusal_probability(self.refusal,np.array([[s1,s1-s2,glob,s1-top5]],np.float32))[0]);accepted=prob>=self.refusal['threshold']
        ids=self.gallery['sample_id'].astype(str);return {'accepted':bool(accepted),'match_probability':prob,'candidates':[{'id':ids[i],'score':s,'ann_score':a} for i,s,a in pairs] if accepted else [],'embedding':z[0].tolist()}
