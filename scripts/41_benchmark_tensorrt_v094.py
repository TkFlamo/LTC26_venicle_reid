#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.data.dataset import _baseline_v4_transform, _crop_source_bbox
from vehicle_fingerprint.data.color import robust_color_descriptor
from vehicle_fingerprint.tensorrt_deploy import TensorRTEngine, weighted_ensemble_embedding, topk_cosine, summarize_times, write_json
from vehicle_fingerprint.pairs import pair_feature_np_cross
from vehicle_fingerprint.refusal import refusal_probability




def _portable_manifest_path(export_dir: Path, value, fallback: Path) -> Path:
    """Resolve an exported absolute path, then fall back to bundle-relative layout."""
    if value:
        q=Path(str(value))
        if q.is_file(): return q
        if not q.is_absolute() and (ROOT/q).is_file(): return ROOT/q
    if fallback.is_file(): return fallback
    raise FileNotFoundError(f"Deployment file not found: {value!r}; fallback={fallback}")

def _resolve_path(v):
    p = Path(str(v))
    return p if p.is_absolute() else ROOT / p


def _open_row(row) -> Image.Image:
    idx = row.index
    source = _resolve_path(row.source_path) if "source_path" in idx and str(row.source_path) not in ("", "nan", "None") else None
    can_bbox = source is not None and source.is_file() and all(k in idx for k in ("x", "y", "w", "h"))
    path = source if can_bbox else _resolve_path(row.path if "path" in idx else row.image_path)
    with Image.open(path) as im:
        im = im.convert("RGB")
        if can_bbox:
            im = _crop_source_bbox(im, row, 0.03)
        return im.copy()


class ManifestTensorDataset(Dataset):
    def __init__(self, manifest: str | Path, image_size=(384, 576), limit: int | None = None):
        self.df = pd.read_csv(manifest)
        if limit and limit > 0:
            self.df = self.df.iloc[: int(limit)].reset_index(drop=True)
        self.transform = _baseline_v4_transform(image_size, False)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        im = _open_row(row)
        x = self.transform(im)
        sid = str(row.image_id) if "image_id" in row.index else str(row.sample_id if "sample_id" in row.index else i)
        return x, sid


def _percentiles_ms(vals):
    return summarize_times([float(v) for v in vals])


def _cuda_timed(fn, stream=None):
    stream = stream or torch.cuda.current_stream()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record(stream)
    out = fn()
    en.record(stream); en.synchronize()
    return out, float(st.elapsed_time(en))


def _bench_single_engine(engine: TensorRTEngine, batches, h, w, warmup, iterations):
    rows=[]
    for b in batches:
        r=engine.benchmark((b,3,h,w),warmup=warmup,iterations=iterations)
        rows.append(r)
    return rows


def _ensemble_once(vit, cn, x, weight, mode="sequential"):
    if mode == "sequential":
        stream=torch.cuda.current_stream(x.device)
        vo=vit.infer(x,stream)["embedding"]
        co=cn.infer(x,stream)["embedding"]
        return weighted_ensemble_embedding(vo,co,weight)
    if mode != "parallel": raise ValueError(mode)
    sv=torch.cuda.Stream(device=x.device); sc=torch.cuda.Stream(device=x.device); control=torch.cuda.current_stream(x.device)
    start=torch.cuda.Event(); dv=torch.cuda.Event(); dc=torch.cuda.Event()
    start.record(control); sv.wait_event(start); sc.wait_event(start)
    with torch.cuda.stream(sv):
        vo=vit.infer(x,sv)["embedding"]; dv.record(sv)
    with torch.cuda.stream(sc):
        co=cn.infer(x,sc)["embedding"]; dc.record(sc)
    control.wait_event(dv); control.wait_event(dc)
    return weighted_ensemble_embedding(vo,co,weight)


def _bench_ensemble(vit, cn, batches, h, w, weight, warmup, iterations, mode):
    rows=[]
    input_dtype = torch.float32
    for b in batches:
        x=torch.randn(b,3,h,w,device=vit.device,dtype=input_dtype)
        for _ in range(warmup): _ensemble_once(vit,cn,x,weight,mode)
        torch.cuda.synchronize()
        vals=[]
        for _ in range(iterations):
            st=torch.cuda.Event(enable_timing=True); en=torch.cuda.Event(enable_timing=True)
            st.record()
            _ensemble_once(vit,cn,x,weight,mode)
            en.record(); en.synchronize(); vals.append(float(st.elapsed_time(en)))
        s=_percentiles_ms(vals); s.update({"batch":b,"images_per_second":float(b*1000.0/s["mean_ms"]),"mode":mode})
        rows.append(s)
    return rows


def _extract_gallery_ensemble(vit, cn, manifest, *, batch, workers, weight, h, w, limit=None, parallel=False):
    ds=ManifestTensorDataset(manifest,(h,w),limit=limit)
    loader=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=workers,pin_memory=True,persistent_workers=workers>0)
    zs=[]; ids=[]
    torch.cuda.synchronize(); t0=time.perf_counter()
    for x,sid in loader:
        x=x.to(vit.device,non_blocking=True)
        z=_ensemble_once(vit,cn,x,weight,"parallel" if parallel else "sequential")
        zs.append(z.detach().float()); ids.extend(list(sid))
    torch.cuda.synchronize(); sec=time.perf_counter()-t0
    z=torch.cat(zs,0)
    return z, ids, {"images":len(ds),"wall_seconds":sec,"images_per_second":len(ds)/max(sec,1e-9),"batch":batch,"parallel":bool(parallel)}


def _bench_actual_queries(vit,cn,query_manifest,gallery_z,*,weight,h,w,count=100,topk=10,parallel=False):
    df=pd.read_csv(query_manifest).iloc[:int(count)].reset_index(drop=True)
    transform=_baseline_v4_transform((h,w),False)
    pre=[];h2d=[];infer=[];retr=[];total=[]
    for _,row in df.iterrows():
        t0=time.perf_counter(); im=_open_row(row); x=transform(im)[None]; pre_ms=(time.perf_counter()-t0)*1000.0
        torch.cuda.synchronize(); t0=time.perf_counter(); x=x.to(vit.device,non_blocking=False); torch.cuda.synchronize(); h2d_ms=(time.perf_counter()-t0)*1000.0
        st=torch.cuda.Event(enable_timing=True); en=torch.cuda.Event(enable_timing=True); st.record(); z=_ensemble_once(vit,cn,x,weight,"parallel" if parallel else "sequential"); en.record(); en.synchronize(); inf_ms=float(st.elapsed_time(en))
        st=torch.cuda.Event(enable_timing=True); en=torch.cuda.Event(enable_timing=True); st.record(); _=topk_cosine(z,gallery_z,topk); en.record(); en.synchronize(); ret_ms=float(st.elapsed_time(en))
        pre.append(pre_ms);h2d.append(h2d_ms);infer.append(inf_ms);retr.append(ret_ms);total.append(pre_ms+h2d_ms+inf_ms+ret_ms)
    return {
        "queries":len(df),"parallel":bool(parallel),
        "preprocess_decode_crop_resize":_percentiles_ms(pre),
        "h2d":_percentiles_ms(h2d),
        "ensemble_tensorrt":_percentiles_ms(infer),
        "similarity_topk":_percentiles_ms(retr),
        "end_to_end":_percentiles_ms(total),
        "queries_per_second_from_mean_e2e":float(1000.0/max(np.mean(total),1e-9)),
    }




def _recipe_embedding_torch(zg: torch.Tensor, zf: torch.Tensor, alpha: float) -> torch.Tensor:
    a=float(np.clip(alpha,0.0,1.0)); zg=torch.nn.functional.normalize(zg.float(),dim=-1); zf=torch.nn.functional.normalize(zf.float(),dim=-1)
    if a<=0:return zg
    if a>=1:return zf
    return torch.cat([math.sqrt(1-a)*zg, math.sqrt(a)*zf],dim=-1)


def _full_cache_from_manifest(full: TensorRTEngine, manifest, *, batch, workers, h, w, limit=None):
    df=pd.read_csv(manifest)
    if limit and limit>0: df=df.iloc[:int(limit)].reset_index(drop=True)
    transform=_baseline_v4_transform((h,w),False)
    rows=[]; tensors=[]; colors=[]; ids=[]
    # Decode/transform here rather than via DataLoader so the exact cropped RGB used for color descriptor is retained.
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _,row in df.iterrows():
        im=_open_row(row); tensors.append(transform(im)); colors.append(robust_color_descriptor(im,None));
        ids.append(str(row.image_id) if "image_id" in row.index else str(len(ids)))
    prep_sec=time.perf_counter()-t0
    outs={k:[] for k in ("z_fused","z_global","z_local","parts","visibility","visibility_score","local")}
    torch.cuda.synchronize(); t0=time.perf_counter()
    for st in range(0,len(tensors),int(batch)):
        x=torch.stack(tensors[st:st+int(batch)]).to(full.device)
        o=full.infer(x); torch.cuda.synchronize()
        for k in outs: outs[k].append(o[k].detach().float().cpu().numpy())
    infer_sec=time.perf_counter()-t0
    cache={k:np.concatenate(v,axis=0) for k,v in outs.items()}
    cache["visibility"]=(cache["visibility"]>=0.5).astype(np.uint8)
    cache["color"]=np.stack(colors).astype(np.float32)
    cache["camera_id"]=np.asarray(["-1"]*len(ids),dtype=str); cache["sample_id"]=np.asarray(ids,dtype=str)
    return cache,{"images":len(ids),"preprocess_seconds":prep_sec,"tensorrt_seconds":infer_sec,"total_seconds":prep_sec+infer_sec,"images_per_second_total":len(ids)/max(prep_sec+infer_sec,1e-9)}


def _bench_full_service_like(full: TensorRTEngine, rer: TensorRTEngine|None, query_manifest, gallery_cache, *, recipe, refusal, h,w,count=50,topk=10):
    df=pd.read_csv(query_manifest).iloc[:int(count)].reset_index(drop=True); transform=_baseline_v4_transform((h,w),False)
    alpha=float(recipe.get("base_alpha",0.0)); beta=float(recipe.get("reranker_beta",0.0)); rk=int(recipe.get("rerank_topk",100))
    gg=torch.from_numpy(gallery_cache["z_global"]).to(full.device); gf=torch.from_numpy(gallery_cache["z_fused"]).to(full.device); gallery_z=_recipe_embedding_torch(gg,gf,alpha)
    stage={k:[] for k in ("preprocess","feature_trt","ann_topk","pair_features_cpu","reranker_trt","refusal_cpu","end_to_end")}
    for _,row in df.iterrows():
        wall0=time.perf_counter(); t=time.perf_counter(); im=_open_row(row); color=robust_color_descriptor(im,None)[None].astype(np.float32); x=transform(im)[None]; stage["preprocess"].append((time.perf_counter()-t)*1000)
        x=x.to(full.device)
        st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();o=full.infer(x);en.record();en.synchronize();stage["feature_trt"].append(float(st.elapsed_time(en)))
        q={k:o[k].detach().float().cpu().numpy() for k in ("z_fused","z_global","z_local","parts","visibility","visibility_score","local")}; q["visibility"]=(q["visibility"]>=0.5).astype(np.uint8);q["color"]=color;q["camera_id"]=np.asarray(["-1"]);q["sample_id"]=np.asarray(["query"])
        qz=_recipe_embedding_torch(o["z_global"],o["z_fused"],alpha)
        st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();vals,inds=torch.topk(qz@gallery_z.T,min(rk,gallery_z.shape[0]),dim=1);en.record();en.synchronize();stage["ann_topk"].append(float(st.elapsed_time(en)))
        inds_np=inds[0].detach().cpu().numpy(); base=vals[0].detach().float().cpu().numpy(); scores=np.clip((base+1.0)*0.5,0,1)
        rr_ms=0.0
        if rer is not None and beta>0 and len(inds_np):
            t=time.perf_counter(); X=np.stack([pair_feature_np_cross(q,0,gallery_cache,int(gi)) for gi in inds_np]).astype(np.float32);stage["pair_features_cpu"].append((time.perf_counter()-t)*1000)
            xt=torch.from_numpy(X).to(full.device); st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();ro=rer.infer(xt);en.record();en.synchronize();rr_ms=float(st.elapsed_time(en));stage["reranker_trt"].append(rr_ms)
            prob=ro.get("match_probability",next(iter(ro.values()))).detach().float().cpu().numpy(); scores=(1-beta)*scores+beta*prob
            order=np.argsort(-scores,kind="stable");inds_np=inds_np[order];scores=scores[order]
        else:
            stage["pair_features_cpu"].append(0.0);stage["reranker_trt"].append(0.0)
        t=time.perf_counter()
        if refusal and len(scores):
            s1=float(scores[0]);s2=float(scores[1]) if len(scores)>1 else 0.0;top5=float(np.mean(scores[:min(5,len(scores))]));gi=int(inds_np[0]); glob=float(np.dot(q["z_global"][0],gallery_cache["z_global"][gi])); X=np.asarray([[s1,s1-s2,glob,s1-top5]],np.float32); _=refusal_probability(refusal,X)[0]
        stage["refusal_cpu"].append((time.perf_counter()-t)*1000)
        stage["end_to_end"].append((time.perf_counter()-wall0)*1000)
    return {k:_percentiles_ms(v) for k,v in stage.items()}|{"queries":len(df),"queries_per_second_from_mean_e2e":1000.0/max(float(np.mean(stage["end_to_end"])),1e-9)}

def _bench_full_engine(full: TensorRTEngine,batches,h,w,warmup,iterations):
    return _bench_single_engine(full,batches,h,w,warmup,iterations)


def _bench_reranker(rer: TensorRTEngine, pairs=(1,10,50,100,200), warmup=50, iterations=200):
    rows=[]
    input_dim=int(rer.engine.get_tensor_shape(rer.input_name)[-1])
    # Dynamic engine shape can report -1; infer input width from optimization profile if needed.
    if input_dim < 0:
        try:
            mn,opt,mx=rer.engine.get_tensor_profile_shape(rer.input_name,0); input_dim=int(opt[-1])
        except Exception:
            raise RuntimeError("Could not infer reranker input_dim from TensorRT engine")
    for n in pairs:
        r=rer.benchmark((int(n),input_dim),warmup=warmup,iterations=iterations); r["pairs"]=r.pop("batch"); rows.append(r)
    return rows


def main():
    p=argparse.ArgumentParser(description="Benchmark v0.9.4 TensorRT global ensemble and optional full neural pipeline")
    p.add_argument("--export-dir",default="deploy/tensorrt_v094")
    p.add_argument("--device",default="0")
    p.add_argument("--batches",default="1,8,16,32")
    p.add_argument("--warmup",type=int,default=50);p.add_argument("--iterations",type=int,default=200)
    p.add_argument("--convnext-weight",type=float,default=0.45,help="Score weight of ConvNeXt; cross-fit folds clustered around 0.45")
    p.add_argument("--query-manifest",default="data/processed/hackathon_single_v5shared/inner/selection_official/query.csv")
    p.add_argument("--gallery-manifest",default="data/processed/hackathon_single_v5shared/inner/selection_official/gallery.csv")
    p.add_argument("--actual-queries",type=int,default=100)
    p.add_argument("--gallery-limit",type=int,default=0)
    p.add_argument("--gallery-batch",type=int,default=32);p.add_argument("--workers",type=int,default=4)
    p.add_argument("--topk",type=int,default=10)
    p.add_argument("--skip-actual",action="store_true")
    p.add_argument("--skip-parallel",action="store_true")
    p.add_argument("--out",default="artifacts/benchmarks/tensorrt_v094")
    a=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA is required")
    device=torch.device(f"cuda:{a.device}"); torch.cuda.set_device(device)
    d=ROOT/a.export_dir if not Path(a.export_dir).is_absolute() else Path(a.export_dir)
    manifest=json.loads((d/"export_manifest.json").read_text(encoding="utf-8"))
    h=int(manifest["shape_profile"]["height"]);w=int(manifest["shape_profile"]["width"])
    e=manifest["models"]
    vit_path=_portable_manifest_path(d,e["vit_global"].get("engine"),d/"engines"/"vit_global.plan")
    cn_path=_portable_manifest_path(d,e["convnext_global"].get("engine"),d/"engines"/"convnext_global.plan")
    vit=TensorRTEngine(vit_path,device);cn=TensorRTEngine(cn_path,device)
    batches=[int(x) for x in a.batches.split(",") if x.strip()]
    batches=[b for b in batches if b<=int(manifest["shape_profile"]["max_batch"])]
    result={
        "schema":"vehicle-reid-v094-tensorrt-benchmark-v1",
        "gpu":torch.cuda.get_device_name(device),"torch":torch.__version__,"convnext_weight":a.convnext_weight,
        "pure_engine":{},"ensemble":{},"actual_pipeline":{},"optional_full":{},
    }
    result["pure_engine"]["vit_global"]=_bench_single_engine(vit,batches,h,w,a.warmup,a.iterations)
    result["pure_engine"]["convnext_global"]=_bench_single_engine(cn,batches,h,w,a.warmup,a.iterations)
    result["ensemble"]["sequential"]=_bench_ensemble(vit,cn,batches,h,w,a.convnext_weight,a.warmup,a.iterations,"sequential")
    if not a.skip_parallel:
        result["ensemble"]["parallel_cuda_streams"]=_bench_ensemble(vit,cn,batches,h,w,a.convnext_weight,a.warmup,a.iterations,"parallel")

    qpath=ROOT/a.query_manifest if not Path(a.query_manifest).is_absolute() else Path(a.query_manifest)
    gpath=ROOT/a.gallery_manifest if not Path(a.gallery_manifest).is_absolute() else Path(a.gallery_manifest)
    if not a.skip_actual and qpath.is_file() and gpath.is_file():
        glimit=a.gallery_limit if a.gallery_limit>0 else None
        gallery_z,_,gb=_extract_gallery_ensemble(vit,cn,gpath,batch=a.gallery_batch,workers=a.workers,weight=a.convnext_weight,h=h,w=w,limit=glimit,parallel=False)
        result["actual_pipeline"]["gallery_build_sequential"]=gb
        result["actual_pipeline"]["query_sequential"]=_bench_actual_queries(vit,cn,qpath,gallery_z,weight=a.convnext_weight,h=h,w=w,count=a.actual_queries,topk=a.topk,parallel=False)
        if not a.skip_parallel:
            result["actual_pipeline"]["query_parallel_cuda_streams"]=_bench_actual_queries(vit,cn,qpath,gallery_z,weight=a.convnext_weight,h=h,w=w,count=a.actual_queries,topk=a.topk,parallel=True)
    else:
        result["actual_pipeline"]["note"]="Actual manifest benchmark skipped or query/gallery manifest not found"

    full=None; rer=None
    if "full_feature_extractor" in e and e["full_feature_extractor"].get("engine"):
        full_path=_portable_manifest_path(d,e["full_feature_extractor"].get("engine"),d/"engines"/"full_feature_extractor.plan")
        full=TensorRTEngine(full_path,device)
        result["optional_full"]["feature_extractor"]=_bench_full_engine(full,batches,h,w,a.warmup,a.iterations)
    if "pair_reranker" in e and e["pair_reranker"].get("engine"):
        rer_path=_portable_manifest_path(d,e["pair_reranker"].get("engine"),d/"engines"/"pair_reranker.plan")
        rer=TensorRTEngine(rer_path,device)
        result["optional_full"]["pair_reranker"]=_bench_reranker(rer,warmup=a.warmup,iterations=a.iterations)
    if full is not None and not a.skip_actual and qpath.is_file() and gpath.is_file():
        fc=e["full_feature_extractor"]; recipe={}
        rp=fc.get("retrieval_recipe"); dep=ROOT/"deploy"/"models_v094_v5exact_base_veri"
        try:
            rpp=_portable_manifest_path(d,rp,dep/"retrieval_recipe.json"); recipe=json.loads(rpp.read_text(encoding="utf-8"))
        except FileNotFoundError: pass
        refusal=None; fp=fc.get("refusal")
        try:
            fpp=_portable_manifest_path(d,fp,dep/"refusal.json"); refusal=json.loads(fpp.read_text(encoding="utf-8"))
        except FileNotFoundError: pass
        gcache,gb=_full_cache_from_manifest(full,gpath,batch=a.gallery_batch,workers=a.workers,h=h,w=w,limit=(a.gallery_limit if a.gallery_limit>0 else None))
        result["optional_full"]["gallery_build"]=gb
        result["optional_full"]["actual_service_like"]=_bench_full_service_like(full,rer,qpath,gcache,recipe=recipe,refusal=refusal,h=h,w=w,count=min(a.actual_queries,50),topk=a.topk)

    out=ROOT/a.out if not Path(a.out).is_absolute() else Path(a.out);out.mkdir(parents=True,exist_ok=True)
    write_json(out/"benchmark.json",result)
    rows=[]
    for section,name_rows in result["pure_engine"].items():
        for r in name_rows:rows.append({"component":section,**r})
    for mode,name_rows in result["ensemble"].items():
        for r in name_rows:rows.append({"component":f"ensemble_{mode}",**r})
    pd.DataFrame(rows).to_csv(out/"benchmark_summary.csv",index=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print(f"\n[DONE] {out/'benchmark.json'}")

if __name__=="__main__":main()
