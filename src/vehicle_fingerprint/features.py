from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.color import robust_color_descriptor
from .data.dataset import ReIDDataset, IMAGENET_MEAN, IMAGENET_STD
from .models.reid import VehicleFingerprintModel
from .utils import resolve_device, autocast_context

META_KEYS={"vehicle_key","camera_id","sample_id","path","orig_size"}


def _string_array(values): return np.asarray([str(x) for x in values],dtype=np.str_)


def _assert_pickle_free_npz(path):
    with np.load(path,allow_pickle=False) as d:
        for k in d.files:_=d[k]




def _inference_worker_init(worker_id: int) -> None:
    """Prevent nested OpenCV/Torch CPU pools inside DataLoader workers."""
    del worker_id
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

def _collate(batch):
    out={}
    for k in batch[0]:out[k]=[x[k] for x in batch] if k in META_KEYS else torch.stack([x[k] for x in batch])
    return out


def _model_from_checkpoint_cfg(mcfg: dict, num_classes: int=0):
    return VehicleFingerprintModel(
        mcfg["backbone"],num_classes=num_classes,embed_dim=mcfg.get("embed_dim",512),part_dim=mcfg.get("part_dim",128),local_dim=mcfg.get("local_dim",128),local_grid=tuple(mcfg.get("local_grid",[4,6])),fusion_layers=mcfg.get("fusion_layers",2),fusion_heads=mcfg.get("fusion_heads",8),arc_scale=mcfg.get("arc_scale",30.),arc_margin=mcfg.get("arc_margin",.35),part_head_dim=mcfg.get("part_head_dim",256),part_head_blocks=mcfg.get("part_head_blocks",3),part_attention_heads=mcfg.get("part_attention_heads",8),part_prior_strength=mcfg.get("part_prior_strength",3.),part_visibility_threshold=mcfg.get("part_visibility_threshold",.42),part_visibility_topk=mcfg.get("part_visibility_topk",3),detach_semantic_prior=mcfg.get("detach_semantic_prior",True),part_dropout=0.,fusion_max_residual=mcfg.get("fusion_max_residual",.25),fusion_initial_gate=mcfg.get("fusion_initial_gate",.01),enable_parts=mcfg.get("enable_parts",True),global_head_mode=mcfg.get("global_head_mode","baseline_v4"),multiscale_spatial=mcfg.get("multiscale_spatial",False),spatial_fusion_dim=mcfg.get("spatial_fusion_dim",256)
    )


def load_inference_model(checkpoint: str|Path,device="auto"):
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False);mcfg=ck["model_cfg"];model=_model_from_checkpoint_cfg(mcfg,0);state={k:v for k,v in ck["model"].items() if not k.startswith("classifier.")};model.load_state_dict(state,strict=False);dev=resolve_device(device);model.to(dev).eval();return model,dev,mcfg


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    mean=torch.tensor(IMAGENET_MEAN,device=x.device,dtype=x.dtype)[:,None,None];std=torch.tensor(IMAGENET_STD,device=x.device,dtype=x.dtype)[:,None,None]
    y=(x*std+mean).clamp(0,1).mul(255).byte().permute(1,2,0).cpu().numpy()
    return Image.fromarray(y)


def extract_feature_cache(manifest,checkpoint,out_npz,*,part_root=None,device="auto",precision="bf16",batch_size=48,workers=8,image_size=None):
    df=pd.read_csv(manifest);tmp=df.copy()
    if "vehicle_id" not in tmp.columns:tmp["vehicle_id"]="unknown"
    if "dataset" not in tmp.columns:tmp["dataset"]="input"
    label_map={k:i for i,k in enumerate(sorted(set(tmp.dataset.astype(str)+":"+tmp.vehicle_id.astype(str))))}
    model,dev,mcfg=load_inference_model(checkpoint,device);prep=mcfg.get("preprocess",{})
    size=image_size if image_size is not None else prep.get("image_size",[256,384])
    ds=ReIDDataset(tmp,image_size=size,train=False,label_map=label_map,augmentation_profile=prep.get("augmentation_profile","baseline_v4"),use_source_bbox=prep.get("use_source_bbox",True),bbox_pad=prep.get("bbox_pad",.03),inference_only=True,return_color_descriptor=True)
    loader_kwargs=dict(batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True,collate_fn=_collate)
    if int(workers)>0:
        loader_kwargs.update(persistent_workers=True,prefetch_factor=2,worker_init_fn=_inference_worker_init)
    loader=DataLoader(ds,**loader_kwargs)
    Zf=[];Zg=[];Zl=[];Parts=[];Vis=[];VisScore=[];Local=[];Colors=[];sids=[];vids=[];cams=[];paths=[]
    with torch.inference_mode():
        for b in tqdm(loader,desc="extract features"):
            x=b["image"].to(dev,non_blocking=True)
            with autocast_context(dev,precision):o=model(x)
            Zf.append(o["z_fused"].float().cpu().numpy());Zg.append(o["z_global"].float().cpu().numpy());Zl.append(o["z_local"].float().cpu().numpy());Parts.append(o["parts"].float().cpu().numpy().astype(np.float16));Vis.append(o["visibility"].cpu().numpy().astype(np.uint8));VisScore.append(o["visibility_score"].float().cpu().numpy().astype(np.float16));Local.append(o["local"].float().cpu().numpy().astype(np.float16))
            if "color" in b:
                Colors.extend(b["color"].float().cpu().numpy())
            else:
                for j in range(len(x)):Colors.append(robust_color_descriptor(_tensor_to_pil(x[j]),None))
            sids.extend(b["sample_id"]);vids.extend(b["vehicle_key"]);cams.extend(b["camera_id"]);paths.extend(b["path"])
    payload={"z_fused":np.concatenate(Zf).astype(np.float32),"z_global":np.concatenate(Zg).astype(np.float32),"z_local":np.concatenate(Zl).astype(np.float32),"parts":np.concatenate(Parts),"visibility":np.concatenate(Vis),"visibility_score":np.concatenate(VisScore),"local":np.concatenate(Local),"color":np.stack(Colors).astype(np.float32),"sample_id":_string_array(sids),"vehicle_key":_string_array(vids),"camera_id":_string_array(cams),"path":_string_array(paths)}
    for col in ("image_id","source_row","dataset","split"):
        if col in df.columns:payload[f"meta_{col}"]=_string_array(df[col].astype(str).tolist())
    out_npz=Path(out_npz);out_npz.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out_npz,**payload);_assert_pickle_free_npz(out_npz);return out_npz


def load_cache(path):
    d=np.load(path,allow_pickle=False);return {k:d[k] for k in d.files}


def concat_feature_caches(paths, out_npz):
    """Concatenate OOF caches produced by mutually exclusive validation folds."""
    caches=[load_cache(p) for p in paths]
    if not caches: raise ValueError("No caches supplied")
    common=set(caches[0])
    for c in caches[1:]: common &= set(c)
    payload={}
    for k in sorted(common):
        vals=[c[k] for c in caches]
        if vals[0].ndim == 0: continue
        payload[k]=np.concatenate(vals,axis=0)
    # OOF sample ids must be unique; leakage here invalidates reranker training.
    if "sample_id" in payload and len(set(payload["sample_id"].astype(str).tolist())) != len(payload["sample_id"]):
        raise ValueError("Duplicate sample_id in OOF cache concatenation")
    out=Path(out_npz);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**payload);_assert_pickle_free_npz(out);return out


def ensemble_feature_caches(cache_a, cache_b, out_npz, *, weight_a: float = 0.5):
    """Create a score-equivalent two-backbone cache by concatenating normalized embeddings.

    dot(concat(sqrt(w) za, sqrt(1-w) zb)) equals the convex similarity ensemble. Local/part
    evidence is retained from the first cache; this keeps the existing pair-reranker interface.
    """
    a=load_cache(cache_a) if not isinstance(cache_a,dict) else cache_a
    b=load_cache(cache_b) if not isinstance(cache_b,dict) else cache_b
    if len(a["sample_id"])!=len(b["sample_id"]) or not np.array_equal(a["sample_id"].astype(str),b["sample_id"].astype(str)):
        raise ValueError("Ensemble caches must have identical sample order/ids")
    w=float(np.clip(weight_a,0,1))
    def nz(x):
        x=x.astype(np.float32,copy=True);x/=np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12);return x
    payload={k:v for k,v in a.items()}
    for key in ("z_global","z_fused","z_local"):
        if key in a and key in b:
            payload[key]=np.concatenate([np.sqrt(w)*nz(a[key]),np.sqrt(1-w)*nz(b[key])],axis=1).astype(np.float32)
    # Preserve score equivalence for semantic part and local-token cosine evidence as well.  Both
    # families use the same slot/local-grid topology, while descriptor channel counts may differ.
    def nz_last(x):
        x=x.astype(np.float32,copy=True);x/=np.linalg.norm(x,axis=-1,keepdims=True).clip(1e-12);return x
    for key in ("parts","local"):
        if key in a and key in b and a[key].shape[:-1]==b[key].shape[:-1]:
            payload[key]=np.concatenate([np.sqrt(w)*nz_last(a[key]),np.sqrt(1-w)*nz_last(b[key])],axis=-1).astype(np.float16)
    if "visibility_score" in a and "visibility_score" in b and a["visibility_score"].shape==b["visibility_score"].shape:
        payload["visibility_score"]=(w*a["visibility_score"].astype(np.float32)+(1-w)*b["visibility_score"].astype(np.float32)).astype(np.float16)
    if "visibility" in a and "visibility" in b and a["visibility"].shape==b["visibility"].shape:
        payload["visibility"]=(a["visibility"].astype(bool)|b["visibility"].astype(bool)).astype(np.uint8)
    out=Path(out_npz);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**payload);_assert_pickle_free_npz(out);return out
