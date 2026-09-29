from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .official_eval import official_selection_key, load_official_evaluator
from .official_validation import evaluate_recipe, _refusal_features


def checkpoint_official_metrics(path: str|Path) -> dict:
    import torch
    ck=torch.load(path,map_location='cpu',weights_only=False);m=ck.get('metrics',{})
    # baseline checkpoints use unprefixed official keys; detail checkpoints use official_val_best_*.
    def get(*names,default=0.0):
        for n in names:
            if n in m:return float(m[n])
        return float(default)
    return {
        'epoch':int(ck.get('epoch',0)),
        'mAP@10':get('official_val_mAP@10','official_val_best_mAP@10'),
        'Rank-1':get('official_val_Rank-1','official_val_best_Rank-1'),
        'Rank-5':get('official_val_Rank-5','official_val_best_Rank-5'),
        'mAP_full':get('official_val_mAP_full','official_val_best_mAP_full'),
        'mINP':get('official_val_mINP','official_val_best_mINP'),
    }


def aggregate_cv_rows(rows: list[dict], group_cols=('backbone','resolution')) -> pd.DataFrame:
    df=pd.DataFrame(rows)
    agg=df.groupby(list(group_cols),dropna=False).agg(
        folds=('fold','nunique'),mAP10_mean=('mAP@10','mean'),mAP10_std=('mAP@10','std'),
        Rank1_mean=('Rank-1','mean'),Rank5_mean=('Rank-5','mean'),mAPfull_mean=('mAP_full','mean'),mINP_mean=('mINP','mean'),
        best_epoch_median=('epoch','median'),
    ).reset_index()
    agg['mAP10_std']=agg['mAP10_std'].fillna(0.0)
    # Primary metric remains mean official mAP@10; std is diagnostic and a tiny tie breaker only.
    return agg.sort_values(['mAP10_mean','Rank1_mean','Rank5_mean','mAPfull_mean','mINP_mean','mAP10_std'],ascending=[False,False,False,False,False,True],kind='stable')


def select_best_by_family(table: pd.DataFrame) -> dict[str,dict]:
    out={}
    for family,g in table.groupby(table['backbone'].astype(str).map(lambda x:'vit' if x.startswith('vit') else ('convnext' if x.startswith('convnext') else 'other'))):
        out[family]=g.iloc[0].to_dict()
    return out


def select_recipe_across_folds(folds: list[dict], *, alpha_grid, beta_grid, kreciprocal_grid, same_camera_grid, rerank_topk=100, kreciprocal_k=20, device='cuda', evaluator_path='official/evaluate.py'):
    rows=[];best=None
    for a in alpha_grid:
      for b in beta_grid:
       for kr in kreciprocal_grid:
        for scf in same_camera_grid:
            mets=[]
            for f in folds:
                if b>0 and not f.get('reranker'): continue
                rep,*_=evaluate_recipe(f['query_cache'],f['gallery_cache'],f['gt'],base_alpha=float(a),reranker_path=f.get('reranker'),reranker_beta=float(b),rerank_topk=rerank_topk,kreciprocal_lambda=float(kr),kreciprocal_k=kreciprocal_k,same_camera_filter=bool(scf),device=device,evaluator_path=evaluator_path)
                r=rep['ranking'];fr=rep['full_ranking'];mets.append([r['mAP@10'],r['Rank-1'],r['Rank-5'],fr['mAP_full'],fr['mINP']])
            if not mets:continue
            arr=np.asarray(mets,float);mean=arr.mean(0);std=float(arr[:,0].std())
            row={'base_alpha':float(a),'reranker_beta':float(b),'kreciprocal_lambda':float(kr),'same_camera_filter':bool(scf),'mAP@10':mean[0],'Rank-1':mean[1],'Rank-5':mean[2],'mAP_full':mean[3],'mINP':mean[4],'mAP@10_std':std}
            rows.append(row);key=tuple(mean.tolist())+(-std,)
            if best is None or key>best[0]:best=(key,row)
    if best is None:raise RuntimeError('No CV recipe evaluated')
    recipe={**{k:best[1][k] for k in ('base_alpha','reranker_beta','kreciprocal_lambda','same_camera_filter')},'rerank_topk':int(rerank_topk),'kreciprocal_k':int(kreciprocal_k),'selection':'mean_official_CV_mAP@10_then_Rank1_Rank5_full_mAP_mINP'}
    return recipe,pd.DataFrame(rows).sort_values(['mAP@10','Rank-1','Rank-5'],ascending=False,kind='stable')


def fit_pooled_refusal(folds: list[dict], recipe: dict, out_json: str|Path, *, device='cuda', evaluator_path='official/evaluate.py', seed=42):
    allX=[];ally=[];fold_payload=[]
    for f in folds:
        rep,ranked,scores,_,_=evaluate_recipe(f['query_cache'],f['gallery_cache'],f['gt'],base_alpha=recipe['base_alpha'],reranker_path=f.get('reranker'),reranker_beta=recipe['reranker_beta'],rerank_topk=recipe['rerank_topk'],kreciprocal_lambda=recipe.get('kreciprocal_lambda',0),kreciprocal_k=recipe.get('kreciprocal_k',20),same_camera_filter=recipe.get('same_camera_filter',False),device=device,evaluator_path=evaluator_path)
        X,y,meta,query,gallery=_refusal_features(f['query_cache'],f['gallery_cache'],ranked,scores,f['gt'],evaluator_path)
        allX.append(X);ally.append(y);fold_payload.append((X,y,meta,query,gallery))
    X=np.concatenate(allX);y=np.concatenate(ally)
    clf=LogisticRegression(class_weight='balanced',max_iter=2000,random_state=int(seed)).fit(X,y)
    mod=load_official_evaluator(evaluator_path);best=None
    offset=0
    probs_all=clf.predict_proba(X)[:,1]
    for t in np.linspace(.01,.99,197):
        TP=FP=FN=TN=fp_open=0;pra=[];offset=0
        for Xf,yf,meta,query,gallery in fold_payload:
            probs=probs_all[offset:offset+len(Xf)];offset+=len(Xf);cand={}
            for (qid,gid),p in zip(meta,probs):
                if p>=t:cand[str(qid)]=[(str(gid),float(p))]
            m=mod.candidate_metrics(query,gallery,cand);TP+=m['TP'];FP+=m['FP'];FN+=m['FN'];TN+=m['TN'];fp_open+=m['n_openset_queries']-m['TN']
            if np.isfinite(m['PR-AUC']):pra.append(float(m['PR-AUC']))
        prec=TP/(TP+FP) if TP+FP else 0.;rec=TP/(TP+FN) if TP+FN else 0.;f1=2*prec*rec/(prec+rec) if prec+rec else 0.;tnr=TN/(TN+fp_open) if TN+fp_open else float('nan');pr=float(np.mean(pra)) if pra else float('nan')
        key=(f1,tnr if np.isfinite(tnr) else -1,pr if np.isfinite(pr) else -1)
        if best is None or key>best[0]:best=(key,float(t),{'TP':TP,'FP':FP,'FN':FN,'TN':TN,'Precision':prec,'Recall':rec,'F1':f1,'TNR':tnr,'PR-AUC':pr})
    spec={'coef':clf.coef_[0].tolist(),'intercept':float(clf.intercept_[0]),'threshold':best[1],'features':['top1','margin12','global_top1','top1_minus_top5mean'],'calibration_metrics':best[2],'retrieval_recipe':recipe,'selection':'pooled_OOF_F1_then_TNR_then_PR-AUC','seed':int(seed)}
    out=Path(out_json);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8');return spec
