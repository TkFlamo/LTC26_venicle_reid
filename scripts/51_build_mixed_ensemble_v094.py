#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, importlib.util, json, shutil, subprocess, sys
from pathlib import Path
import pandas as pd
import torch, yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vehicle_fingerprint.cv_experiments import checkpoint_official_metrics, fit_pooled_refusal, select_recipe_across_folds
from vehicle_fingerprint.features import ensemble_feature_caches
from vehicle_fingerprint.models.backbone_registry import profile_config
from vehicle_fingerprint.official_validation import evaluate_recipe, generate_official_artifacts, run_official_script
from vehicle_fingerprint.v5_checkpoint import convert_v5_checkpoint


def loadmod(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
P=loadmod('v094_pipeline',ROOT/'scripts/30_full_cv_pipeline.py')
F=loadmod('v094_full_ensemble',ROOT/'scripts/49_full_ensemble_v094.py')


def ry(p):
    with open(p,'r',encoding='utf-8') as f:return yaml.safe_load(f)
def rooted(x):
    p=Path(x).expanduser(); return p if p.is_absolute() else ROOT/p

def prep_external(src:Path,out:Path,backbone='convnext_small'):
    src=src.resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    raw=torch.load(src,map_location='cpu',weights_only=False)
    if isinstance(raw,dict) and 'model_state' in raw and 'model_name' in raw: convert_v5_checkpoint(src,out)
    elif isinstance(raw,dict) and 'model' in raw and 'model_cfg' in raw: shutil.copy2(src,out)
    else: raise RuntimeError(f'Unknown external checkpoint format: {src}')
    ck=torch.load(out,map_location='cpu',weights_only=False); cfg=copy.deepcopy(ck.get('model_cfg',{}))
    exp=str(profile_config(backbone)['model_name']); got=str(cfg.get('backbone',{}).get('model_name',''))
    if got!=exp: raise RuntimeError(f'Expected {exp}, got {got}')
    cfg.setdefault('preprocess',{})['image_size']=[384,576]; ck['model_cfg']=cfg; ck.setdefault('preprocess',{})['image_size']=[384,576]
    torch.save(ck,out); return out

def gs(checkpoint,backbone,initialization,init_checkpoint=None,veri_project=None,external=None):
    try:m=checkpoint_official_metrics(checkpoint)
    except Exception:m={}
    return {'backbone':backbone,'family':'convnext' if backbone.startswith('convnext') else 'vit','resolution':'384x576',
            'initialization':initialization,'init_checkpoint':str(init_checkpoint) if init_checkpoint else None,
            'veri_pretrain_checkpoint':str(veri_project) if veri_project else None,'external_checkpoint':str(external) if external else None,
            'epoch':int(m.get('epoch',0)),'mAP@10':float(m.get('mAP@10',0) or 0),'Rank-1':float(m.get('Rank-1',0) or 0),
            'Rank-5':float(m.get('Rank-5',0) or 0),'mAP_full':float(m.get('mAP_full',0) or 0),'mINP':float(m.get('mINP',0) or 0),
            'checkpoint':str(checkpoint),'native_checkpoint':None}

def bp(pipe,root):
    q=copy.deepcopy(pipe); q['paths']['runs']=str(root); q['paths']['deploy']=str(root/'_unused_deploy'); return q

def ext(pipe,args,manifest,ckpt,out,log):
    P.run_cmd(P.extract_cmd(manifest,ckpt,out,pipe),log=log,marker=out,resume=args.resume,dry=False,keep_going=False); return out

def selrep(summary): return Path(F._selected_representation(summary)['checkpoint'])

def rep_stages(summary):
    """Return neural representation stages only; branch-local listwise reranker is intentionally excluded."""
    out={}
    for x in summary.get('stages',[]):
        st=str(x.get('stage'))
        if st in {'v5_exact_global','spatial_part_aware','detail_tuning'}:
            out[st]=x
    missing={'v5_exact_global','spatial_part_aware','detail_tuning'}-set(out)
    if missing: raise RuntimeError(f'Missing branch representation stages: {sorted(missing)}')
    return out

def stage_weight_search(pipe,args,cn_summary,vi_summary,root,grid):
    """Select branch representation stage pair + ConvNeXt weight + base alpha on development only."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True); gt=Path(pipe['paths']['cv'])/'inner/selection_official/ground_truth.csv'; alpha_grid=list(pipe['reranker'].get('alpha_grid',[0.0]))
    cs=rep_stages(cn_summary); vs=rep_stages(vi_summary); rows=[]
    for cstage,ce in cs.items():
      cq,cg=Path(ce['query_cache']),Path(ce['gallery_cache'])
      for vstage,ve in vs.items():
        vq,vg=Path(ve['query_cache']),Path(ve['gallery_cache'])
        for w in grid:
          d=root/f'{cstage}__{vstage}__w{w:.3f}';d.mkdir(parents=True,exist_ok=True);q,g=d/'q.npz',d/'g.npz'
          if not(args.resume and q.exists()):ensemble_feature_caches(cq,vq,q,weight_a=w)
          if not(args.resume and g.exists()):ensemble_feature_caches(cg,vg,g,weight_a=w)
          for alpha in alpha_grid:
            rep,*_=evaluate_recipe(q,g,gt,base_alpha=float(alpha),device='cuda',evaluator_path='official/evaluate.py');r=rep['ranking'];fr=rep['full_ranking']
            rows.append({'convnext_stage':cstage,'vit_stage':vstage,'convnext_weight':float(w),'vit_weight':float(1-w),'base_alpha':float(alpha),'mAP@10':float(r['mAP@10']),'Rank-1':float(r['Rank-1']),'Rank-5':float(r['Rank-5']),'mAP_full':float(fr['mAP_full']),'mINP':float(fr['mINP'])})
    t=pd.DataFrame(rows).sort_values(['mAP@10','Rank-1','Rank-5','mAP_full','mINP'],ascending=False,kind='stable');t.to_csv(root/'stage_weight_search.csv',index=False);best=t.iloc[0].to_dict();(root/'selected_stage_weight.json').write_text(json.dumps(best,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nSTAGE/WEIGHT SEARCH (development only)\n',t.head(30).to_string(index=False));print('\nSELECTED',json.dumps(best,ensure_ascii=False,indent=2))
    return best

def weight_search(pipe,args,cn,vi,root,grid):
    sd=Path(pipe['paths']['cv'])/'inner/selection_official'; root.mkdir(parents=True,exist_ok=True)
    cq,cg,vq,vg=[root/x for x in ('cn_q.npz','cn_g.npz','vi_q.npz','vi_g.npz')]
    ext(pipe,args,sd/'query.csv',cn,cq,root/'cn.log'); ext(pipe,args,sd/'gallery.csv',cn,cg,root/'cn.log')
    ext(pipe,args,sd/'query.csv',vi,vq,root/'vi.log'); ext(pipe,args,sd/'gallery.csv',vi,vg,root/'vi.log')
    rows=[]
    for w in grid:
        d=root/f'w{w:.3f}'; d.mkdir(exist_ok=True); q,g=d/'q.npz',d/'g.npz'
        if not(args.resume and q.exists()): ensemble_feature_caches(cq,vq,q,weight_a=w)
        if not(args.resume and g.exists()): ensemble_feature_caches(cg,vg,g,weight_a=w)
        rep,*_=evaluate_recipe(q,g,sd/'ground_truth.csv',base_alpha=0,device='cuda',evaluator_path='official/evaluate.py'); r=rep['ranking']; fr=rep['full_ranking']
        rows.append({'convnext_weight':w,'vit_weight':1-w,'mAP@10':r['mAP@10'],'Rank-1':r['Rank-1'],'Rank-5':r['Rank-5'],'mAP_full':fr['mAP_full'],'mINP':fr['mINP']})
    t=pd.DataFrame(rows).sort_values(['mAP@10','Rank-1','Rank-5','mAP_full','mINP'],ascending=False,kind='stable'); t.to_csv(root/'weight_search.csv',index=False)
    w=float(t.iloc[0].convnext_weight); print('\nWEIGHT SEARCH (development only)\n',t.to_string(index=False)); print('selected ConvNeXt weight',w); return w

def fit_mixed_ensemble_dev(pipe,args,cn_summary,vit_summary,root,convnext_weight):
    """Fit ensemble-specific reranker/recipe/refusal using DEVELOPMENT ONLY.

    Unlike the original conservative helper, this searches the configured alpha_grid as well,
    so z_fused / parts / local evidence can contribute to retrieval before the reranker.
    """
    root=Path(root); inner=Path(pipe['paths']['cv'])/'inner'; rr=pipe['reranker']; root.mkdir(parents=True,exist_ok=True)
    cn_ckpt=selrep(cn_summary); vi_ckpt=selrep(vit_summary)

    # Dedicated reranker-fit identities.
    rf=root/'reranker_fit'; rf.mkdir(parents=True,exist_ok=True)
    cn=rf/'convnext.npz'; vi=rf/'vit.npz'; ens=rf/'ensemble.npz'
    ext(pipe,args,inner/'reranker_fit.csv',cn_ckpt,cn,rf/'features_cn.log')
    ext(pipe,args,inner/'reranker_fit.csv',vi_ckpt,vi,rf/'features_vit.log')
    if not(args.resume and ens.exists()): ensemble_feature_caches(cn,vi,ens,weight_a=convnext_weight)
    hard=rf/'hard.json'
    P.run_cmd([sys.executable,'scripts/12_mine_hard_negatives.py','--cache',str(ens),'--out',str(hard),'--topk','30','--representation','global','--refine-factor','3','--top-pair-mean','2'],log=rf/'mine.log',marker=hard,resume=args.resume,dry=False,keep_going=False)
    pairs=rf/'pairs.npz'
    P.run_cmd([sys.executable,'scripts/28_build_reranker_pairs.py','--cache',str(ens),'--hard-map',str(hard),'--out',str(pairs),'--mode',str(rr['mode']),'--candidates-per-query',str(rr['candidates_per_query']),'--groups-per-id',str(rr['groups_per_id']),'--knn-k',str(rr['knn_k'])],log=rf/'pairs.log',marker=pairs,resume=args.resume,dry=False,keep_going=False)
    reranker=root/'reranker'/'best.pt'
    P.run_cmd([sys.executable,'scripts/29_train_reranker_from_pairs.py','--pairs',str(pairs),'--out',str(reranker),'--epochs',str(rr['epochs']),'--device','cuda','--mode',str(rr['mode'])],log=root/'reranker'/'train.log',marker=reranker,resume=args.resume,dry=False,keep_going=False)

    # Separate development selection identities for retrieval recipe + refusal.
    sd=inner/'selection_official'; ss=root/'selection'; ss.mkdir(parents=True,exist_ok=True)
    cq,cg,vq,vg=[ss/x for x in ('cn_q.npz','cn_g.npz','vi_q.npz','vi_g.npz')]
    q,g=ss/'ensemble_q.npz',ss/'ensemble_g.npz'
    ext(pipe,args,sd/'query.csv',cn_ckpt,cq,ss/'features_cn.log'); ext(pipe,args,sd/'gallery.csv',cn_ckpt,cg,ss/'features_cn.log')
    ext(pipe,args,sd/'query.csv',vi_ckpt,vq,ss/'features_vit.log'); ext(pipe,args,sd/'gallery.csv',vi_ckpt,vg,ss/'features_vit.log')
    if not(args.resume and q.exists()): ensemble_feature_caches(cq,vq,q,weight_a=convnext_weight)
    if not(args.resume and g.exists()): ensemble_feature_caches(cg,vg,g,weight_a=convnext_weight)
    desc=[{'query_cache':str(q),'gallery_cache':str(g),'gt':str(sd/'ground_truth.csv'),'reranker':str(reranker)}]
    recipe,search=select_recipe_across_folds(
        desc,
        alpha_grid=rr.get('alpha_grid',[0.0]),
        beta_grid=rr['beta_grid'],
        kreciprocal_grid=rr['kreciprocal_lambda_grid'],
        same_camera_grid=rr['same_camera_filter_grid'],
        rerank_topk=rr['rerank_topk'],
        kreciprocal_k=rr['kreciprocal_k'],
        device='cuda',
    )
    search.to_csv(root/'retrieval_recipe_search.csv',index=False)
    (root/'retrieval_recipe.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2),encoding='utf-8')
    refusal=root/'refusal.json'; fit_pooled_refusal(desc,recipe,refusal,device='cuda')
    rep,*_=evaluate_recipe(q,g,sd/'ground_truth.csv',base_alpha=recipe['base_alpha'],reranker_path=reranker,reranker_beta=recipe['reranker_beta'],rerank_topk=recipe['rerank_topk'],kreciprocal_lambda=recipe['kreciprocal_lambda'],kreciprocal_k=recipe['kreciprocal_k'],same_camera_filter=recipe['same_camera_filter'],device='cuda',evaluator_path='official/evaluate.py')
    r=rep['ranking']; fr=rep['full_ranking']
    out={
        'family':'ensemble','selected_stage':'mixed_full_ensemble_reranker_kreciprocal',
        'convnext_weight':float(convnext_weight),'vit_weight':float(1-convnext_weight),
        'recipe':recipe,'reranker':str(reranker),'refusal':str(refusal),
        'development_selection_metrics':{'mAP@10':float(r['mAP@10']),'Rank-1':float(r['Rank-1']),'Rank-5':float(r['Rank-5']),'mAP_full':float(fr['mAP_full']),'mINP':float(fr['mINP'])},
        'convnext':cn_summary,'vit':vit_summary,
        'selection_leakage_policy':'ensemble weight, alpha/beta/k-reciprocal, reranker and refusal selected on development only; shared validation reporting-only',
    }
    (root/'ensemble_development_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nMIXED ENSEMBLE DEVELOPMENT RECIPE\n',json.dumps({'weight':convnext_weight,'recipe':recipe,'metrics':out['development_selection_metrics']},ensure_ascii=False,indent=2))
    return out


def fit_final_external(pipe,args,summary):
    """Final refit for an already target-trained external global checkpoint.

    Never reruns the project's global V5 trainer. The external ConvNeXt global weights are
    preserved; only the selected downstream spatial/detail stages are refit on all dev rows.
    """
    root=Path(pipe['paths']['runs'])/'final_fit'; cv=Path(pipe['paths']['cv']); b=summary['backbone']; res=[384,576]; cfgdir=Path(pipe['paths']['runs'])/'generated_configs/final'
    base=Path(summary['global_checkpoint'])
    if not base.is_file(): raise FileNotFoundError(base)
    print('[FINAL external] preserving external global checkpoint:',base)
    stage=summary['selected_stage']
    if stage=='v5_exact_global': return {'checkpoint':str(base),'stage':stage,'backbone':b,'resolution':'384x576'}
    pr=root/'parts'; pc=P.part_cfg(pipe['parts']['template'],run_dir=pr,warmstart=base,resolution=res,pipe=pipe,backbone=b); pcp=P.dump_yaml(pc,cfgdir/'parts.yaml')
    P.run_cmd([sys.executable,'scripts/05_pretrain_dino_part_head.py',str(pcp),'--run-dir',str(pr),'--warmstart',str(base),'--expect-backbone',b],log=pr/'train.log',marker=pr/'best.pt',resume=args.resume,dry=False,keep_going=False)
    pa=root/'part_aware'; pac=P.reid_cfg(pipe['part_aware']['template'],train=cv/'dev.csv',val=None,official=None,run_dir=pa,warmstart=pr/'best.pt',resolution=res,pipe=pipe,section='part_aware',backbone=b,no_eval=True,epochs_override=int(summary['part_aware_best_epoch'])); pacp=P.dump_yaml(pac,cfgdir/'partaware.yaml')
    P.run_cmd([sys.executable,'scripts/10_train_reid.py',str(pacp),'--backbone',b,'--run-dir',str(pa),'--warmstart',str(pr/'best.pt'),'--hard-negative-map','none'],log=pa/'train.log',marker=pa/'last.pt',resume=args.resume,dry=False,keep_going=False)
    if stage=='spatial_part_aware': return {'checkpoint':str(pa/'last.pt'),'stage':stage,'backbone':b,'resolution':'384x576'}
    cache=pa/'dev_features.npz'; P.run_cmd(P.extract_cmd(cv/'dev.csv',pa/'last.pt',cache,pipe),log=pa/'features.log',marker=cache,resume=args.resume,dry=False,keep_going=False)
    hard=pa/'hard.json'; P.run_cmd([sys.executable,'scripts/12_mine_hard_negatives.py','--cache',str(cache),'--out',str(hard),'--topk','40','--representation','blend','--alpha','0.15'],log=pa/'mine.log',marker=hard,resume=args.resume,dry=False,keep_going=False)
    dt=root/'detail'; dc=P.reid_cfg(pipe['detail']['template'],train=cv/'dev.csv',val=None,official=None,run_dir=dt,warmstart=pa/'last.pt',resolution=res,pipe=pipe,section='detail',backbone=b,teacher=pr/'best.pt',hard=hard,no_eval=True,epochs_override=int(summary['detail_best_epoch'])); dcp=P.dump_yaml(dc,cfgdir/'detail.yaml')
    P.run_cmd([sys.executable,'scripts/10_train_reid.py',str(dcp),'--backbone',b,'--run-dir',str(dt),'--warmstart',str(pa/'last.pt'),'--teacher-checkpoint',str(pr/'best.pt'),'--hard-negative-map',str(hard)],log=dt/'train.log',marker=dt/'last.pt',resume=args.resume,dry=False,keep_going=False)
    return {'checkpoint':str(dt/'last.pt'),'stage':stage,'backbone':b,'resolution':'384x576'}


def fit_final_veri(pipe,args,summary):
    """Final refit for a VeRi-initialized target ViT branch."""
    root=Path(pipe['paths']['runs'])/'final_fit'; cv=Path(pipe['paths']['cv']); b=summary['backbone']; res=[384,576]; cfgdir=Path(pipe['paths']['runs'])/'generated_configs/final'
    init=summary.get('init_checkpoint')
    if not init or not Path(init).is_file():
        raise RuntimeError(f"VeRi final refit requires valid init_checkpoint, got {init!r}")
    ok,base=P._run_v5_exact_baseline(pipe,args,backbone=b,train_manifest=cv/'dev.csv',val_manifest=None,run_dir=root/'baseline_v5_exact',epochs=int(summary['global_best_epoch']),final_refit=True,init_checkpoint=init)
    if not ok or not Path(base).is_file(): raise RuntimeError('VeRi target global final refit failed')
    stage=summary['selected_stage']
    if stage=='v5_exact_global': return {'checkpoint':str(base),'stage':stage,'backbone':b,'resolution':'384x576'}
    pr=root/'parts'; pc=P.part_cfg(pipe['parts']['template'],run_dir=pr,warmstart=base,resolution=res,pipe=pipe,backbone=b); pcp=P.dump_yaml(pc,cfgdir/'parts.yaml')
    P.run_cmd([sys.executable,'scripts/05_pretrain_dino_part_head.py',str(pcp),'--run-dir',str(pr),'--warmstart',str(base),'--expect-backbone',b],log=pr/'train.log',marker=pr/'best.pt',resume=args.resume,dry=False,keep_going=False)
    pa=root/'part_aware'; pac=P.reid_cfg(pipe['part_aware']['template'],train=cv/'dev.csv',val=None,official=None,run_dir=pa,warmstart=pr/'best.pt',resolution=res,pipe=pipe,section='part_aware',backbone=b,no_eval=True,epochs_override=int(summary['part_aware_best_epoch'])); pacp=P.dump_yaml(pac,cfgdir/'partaware.yaml')
    P.run_cmd([sys.executable,'scripts/10_train_reid.py',str(pacp),'--backbone',b,'--run-dir',str(pa),'--warmstart',str(pr/'best.pt'),'--hard-negative-map','none'],log=pa/'train.log',marker=pa/'last.pt',resume=args.resume,dry=False,keep_going=False)
    if stage=='spatial_part_aware': return {'checkpoint':str(pa/'last.pt'),'stage':stage,'backbone':b,'resolution':'384x576'}
    cache=pa/'dev_features.npz'; P.run_cmd(P.extract_cmd(cv/'dev.csv',pa/'last.pt',cache,pipe),log=pa/'features.log',marker=cache,resume=args.resume,dry=False,keep_going=False)
    hard=pa/'hard.json'; P.run_cmd([sys.executable,'scripts/12_mine_hard_negatives.py','--cache',str(cache),'--out',str(hard),'--topk','40','--representation','blend','--alpha','0.15'],log=pa/'mine.log',marker=hard,resume=args.resume,dry=False,keep_going=False)
    dt=root/'detail'; dc=P.reid_cfg(pipe['detail']['template'],train=cv/'dev.csv',val=None,official=None,run_dir=dt,warmstart=pa/'last.pt',resolution=res,pipe=pipe,section='detail',backbone=b,teacher=pr/'best.pt',hard=hard,no_eval=True,epochs_override=int(summary['detail_best_epoch'])); dcp=P.dump_yaml(dc,cfgdir/'detail.yaml')
    P.run_cmd([sys.executable,'scripts/10_train_reid.py',str(dcp),'--backbone',b,'--run-dir',str(dt),'--warmstart',str(pa/'last.pt'),'--teacher-checkpoint',str(pr/'best.pt'),'--hard-negative-map',str(hard)],log=dt/'train.log',marker=dt/'last.pt',resume=args.resume,dry=False,keep_going=False)
    return {'checkpoint':str(dt/'last.pt'),'stage':stage,'backbone':b,'resolution':'384x576'}

def aggregate(reports):
    rows=[]
    for i,x in enumerate(reports):
        r=x.get('ranking',{});f=x.get('full_ranking',{});c=x.get('candidates',{}); rows.append({'fold':i,'mAP@10':r.get('mAP@10'),'Rank-1':r.get('Rank-1'),'Rank-5':r.get('Rank-5'),'mAP_full':f.get('mAP_full'),'mINP':f.get('mINP'),'Precision':c.get('Precision'),'Recall':c.get('Recall'),'F1':c.get('F1'),'TNR':c.get('TNR'),'PR-AUC':c.get('PR-AUC')})
    df=pd.DataFrame(rows); out={'folds':len(rows),'per_fold':rows}
    for m in ['mAP@10','Rank-1','Rank-5','mAP_full','mINP','Precision','Recall','F1','TNR','PR-AUC']:
        v=pd.to_numeric(df[m],errors='coerce').dropna(); out[m+'_mean']=float(v.mean()) if len(v) else float('nan'); out[m+'_std']=float(v.std(ddof=0)) if len(v) else float('nan')
    return df,out

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--config',default='configs/full_cv_pipeline.yaml'); a.add_argument('--data-project',required=True)
    a.add_argument('--convnext-external',required=True); a.add_argument('--vit-project',required=True); a.add_argument('--vit-veri-init',required=True); a.add_argument('--vit-veri-project',default=None)
    a.add_argument('--convnext-weight-grid',default='0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00')
    a.add_argument('--run-root',default='runs/full_mixed_ensemble_v094'); a.add_argument('--deploy',default='deploy/models_v094_mixed_ensemble'); a.add_argument('--no-resume',action='store_true'); a.add_argument('--check-only',action='store_true',help='Validate paths/checkpoint architectures/split fingerprints, convert external checkpoint, then stop')
    a=a.parse_args(); a.resume=not a.no_resume; a.dry=False; a.keep_going=False
    if not (ROOT/'scripts/49_full_ensemble_v094.py').is_file(): raise RuntimeError('Apply v094_clean_full_ensemble_hotfix first: scripts/49_full_ensemble_v094.py is required')
    pipe=ry(rooted(a.config)); old=Path(a.data_project).expanduser().resolve(); rr=rooted(a.run_root); rr.mkdir(parents=True,exist_ok=True); dep=rooted(a.deploy)
    pipe['paths']['raw_images']=str(old/'raw/images'); pipe['paths']['cv']=str(old/'data/processed/hackathon_single_v5shared'); pipe['paths']['part_bootstrap']=str(old/'data/processed/part_bootstrap'); pipe['paths']['runs']=str(rr); pipe['paths']['deploy']=str(dep)
    subprocess.run([sys.executable,str(ROOT/'scripts/25b_verify_shared_validation.py'),'--cv-root',pipe['paths']['cv'],'--spec',str(pipe['cv']['shared_validation_spec'])],cwd=ROOT,check=True)
    cn_src=Path(a.convnext_external).expanduser().resolve(); vp=Path(a.vit_project).expanduser().resolve(); vi=Path(a.vit_veri_init).expanduser().resolve(); vpp=Path(a.vit_veri_project).expanduser().resolve() if a.vit_veri_project else None
    for p in (cn_src,vp,vi):
        if not p.is_file(): raise FileNotFoundError(p)
    cn=prep_external(cn_src,rr/'inputs/convnext_small_external_project.pt')
    vck=torch.load(vp,map_location='cpu',weights_only=False); got=str(vck.get('model_cfg',{}).get('backbone',{}).get('model_name','')); exp=str(profile_config('vit_base')['model_name'])
    if got!=exp: raise RuntimeError(f'--vit-project must be vit_base project checkpoint: {got}')
    if a.check_only:
        print(json.dumps({'status':'OK','convnext_external':str(cn_src),'convnext_project':str(cn),'vit_project':str(vp),'vit_veri_init':str(vi),'data_project':str(old),'cv_root':pipe['paths']['cv'],'part_bootstrap':pipe['paths']['part_bootstrap']},ensure_ascii=False,indent=2))
        return
    csum=P.advanced_single(bp(pipe,rr/'branches/convnext_small'),a,gs(cn,'convnext_small','external_global',external=cn_src))
    vs=gs(vp,'vit_base','veri776_v5',init_checkpoint=vi,veri_project=vpp)
    if vs['epoch']<=0: raise RuntimeError('Cannot read target ViT best epoch; use global_v5_exact_from_veri/.../project_best.pt')
    vsum=P.advanced_single(bp(pipe,rr/'branches/vit_base_veri'),a,vs)
    grid=sorted(set(float(x) for x in a.convnext_weight_grid.split(','))); choice=stage_weight_search(pipe,a,csum,vsum,rr/'ensemble_stage_weight_selection',grid); w=float(choice['convnext_weight'])
    csum_final=copy.deepcopy(csum); vsum_final=copy.deepcopy(vsum); csum_final['selected_stage']=str(choice['convnext_stage']); vsum_final['selected_stage']=str(choice['vit_stage'])
    ens=fit_mixed_ensemble_dev(pipe,a,csum_final,vsum_final,rr/'ensemble_development',w)
    ens['selected_branch_stages']={'convnext_small':csum_final['selected_stage'],'vit_base':vsum_final['selected_stage']}
    (rr/'ensemble_development'/'ensemble_development_summary.json').write_text(json.dumps(ens,ensure_ascii=False,indent=2),encoding='utf-8')
    cfit=fit_final_external(bp(pipe,rr/'final_refit/convnext_small'),a,csum_final); vfit=fit_final_veri(bp(pipe,rr/'final_refit/vit_base_veri'),a,vsum_final)
    shared=rr/'final_shared_validation'; shared.mkdir(parents=True,exist_ok=True); refusal=json.loads(Path(ens['refusal']).read_text()); reports=[]
    for i,fd in enumerate(P._shared_validation_fold_paths(pipe)):
        fo=shared/f'fold_{i}'; fo.mkdir(parents=True,exist_ok=True); cq,cg,vq,vg=[fo/x for x in ('cn_q.npz','cn_g.npz','vi_q.npz','vi_g.npz')]; q,g=fo/'ensemble_q.npz',fo/'ensemble_g.npz'
        ext(pipe,a,fd/'query.csv',Path(cfit['checkpoint']),cq,fo/'cn.log'); ext(pipe,a,fd/'gallery.csv',Path(cfit['checkpoint']),cg,fo/'cn.log'); ext(pipe,a,fd/'query.csv',Path(vfit['checkpoint']),vq,fo/'vi.log'); ext(pipe,a,fd/'gallery.csv',Path(vfit['checkpoint']),vg,fo/'vi.log')
        if not(a.resume and q.exists()): ensemble_feature_caches(cq,vq,q,weight_a=w)
        if not(a.resume and g.exists()): ensemble_feature_caches(cg,vg,g,weight_a=w)
        gt=fd/'ground_truth.csv'; generate_official_artifacts(q,g,gt,fo,ens['recipe'],refusal,reranker_path=ens['reranker'],device='cuda',evaluator_path='official/evaluate.py'); reports.append(run_official_script(gt,fo/'submission.csv',candidates=fo/'candidates.csv',embeddings=fo/'embeddings.npy',query_csv=fd/'query.csv',gallery_csv=fd/'gallery.csv',json_out=fo/'official_report.json',evaluator_path='official/evaluate.py'))
    df,rep=aggregate(reports); df.to_csv(shared/'shared_validation_metrics.csv',index=False); rep.update({'selected_backbones':['convnext_small','vit_base'],'convnext_weight':w,'vit_weight':1-w,'initializations':{'convnext_small':'external_global','vit_base':'veri776_v5_then_target'},'selected_branch_stages':ens.get('selected_branch_stages'),'selection_leakage_policy':'weight/reranker/recipe/refusal selected on development only; shared reporting-only','hidden_hackathon_test_used':False}); (shared/'shared_validation_report.json').write_text(json.dumps(rep,ensure_ascii=False,indent=2))
    F._export_deployment(pipe,a,cn_fit=cfit,vit_fit=vfit,ensemble_summary=ens,shared_dir=shared,deploy=dep)
    # Rename generic ensemble files so the deployment is self-describing.
    generic_cn=dep/'reid_convnext.pt'; named_cn=dep/'reid_convnext_small.pt'
    generic_vi=dep/'reid_vit.pt'; named_vi=dep/'reid_vit_base.pt'
    if generic_cn.is_file(): generic_cn.replace(named_cn)
    if generic_vi.is_file(): generic_vi.replace(named_vi)
    meta=json.loads((dep/'deployment.json').read_text())
    meta.update({
        'checkpoint_a':'reid_convnext_small.pt',
        'checkpoint_b':'reid_vit_base.pt',
        'checkpoint_a_backbone':'convnext_small',
        'checkpoint_b_backbone':'vit_base',
        'initializations':{'convnext_small':'external_global','vit_base':'veri776_v5_then_target'},
        'selected_branch_stages':ens.get('selected_branch_stages'),
        'convnext_weight':w,'vit_weight':1-w,'weight_a':w,
    })
    (dep/'deployment.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print('\nMIXED FULL ENSEMBLE\n',json.dumps(rep,ensure_ascii=False,indent=2)); print('[DEPLOY]',dep)
if __name__=='__main__': main()
