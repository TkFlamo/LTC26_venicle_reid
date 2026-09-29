from __future__ import annotations

from pathlib import Path
import copy
import torch

from .models.backbone_registry import profile_config


def _profile_from_model_name(model_name: str) -> str:
    for name in ("convnext_large", "vit_large", "convnext_base", "convnext_small", "vit_base", "vit_small"):
        if profile_config(name)["model_name"] == str(model_name):
            return name
    return str(model_name)


def map_v5_state_to_project(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map only the mathematically shared V5 global branch into project names."""
    out={}
    for k,v in state.items():
        if k.startswith("backbone."):
            out["backbone.model."+k[len("backbone."):]]=v
        elif k.startswith("proj."):
            out["global_proj."+k[len("proj."):]]=v
        elif k.startswith("bn."):
            out["global_bn."+k[len("bn."):]]=v
        elif k.startswith("cls."):
            out["classifier."+k[len("cls."):]]=v
    return out


def convert_v5_checkpoint(src: str|Path, dst: str|Path, *, multiscale_spatial: bool=True, spatial_fusion_dim: int=256) -> Path:
    src=Path(src);dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    ck=torch.load(src,map_location="cpu",weights_only=False)
    model_name=str(ck["model_name"]);profile=_profile_from_model_name(model_name);prof=profile_config(profile)
    h,w=int(ck.get("height",384)),int(ck.get("width",576))
    bcfg={
        "profile": profile,
        "model_name": model_name,
        "pretrained": False,
        "checkpoint_path": None,
        "dense_stage": prof.get("dense_stage"),
    }
    if prof.get("dense_stages") is not None: bcfg["dense_stages"]=list(prof["dense_stages"])
    if prof.get("vit_dense_blocks") is not None: bcfg["vit_dense_blocks"]=list(prof["vit_dense_blocks"])
    if str(profile).startswith("vit"):
        bcfg.update({"img_size":[h,w],"dynamic_img_size":True,"dynamic_img_pad":True,"global_feature_mode":"v5_avg"})
    else:
        bcfg["global_feature_mode"]="timm_prelogits"
    args=ck.get("args",{}) or {}
    model_cfg={
        "backbone":bcfg,
        "embed_dim":int(ck.get("embed_dim",512)),
        "part_dim":128,"local_dim":128,"local_grid":[4,6],
        "fusion_layers":2,"fusion_heads":8,
        "arc_scale":float(args.get("arc_s",30.0)),"arc_margin":float(args.get("arc_m",0.35)),
        "part_head_dim":256,"part_head_blocks":2,"part_attention_heads":8,
        "part_prior_strength":2.5,"part_visibility_threshold":0.35,"part_visibility_topk":3,
        "detach_semantic_prior":True,"part_dropout":0.08,
        "fusion_max_residual":0.25,"fusion_initial_gate":0.01,
        "enable_parts":False,"global_head_mode":"v5_exact",
        "multiscale_spatial":bool(multiscale_spatial),"spatial_fusion_dim":int(spatial_fusion_dim),
        "preprocess":{"image_size":[h,w],"augmentation_profile":"baseline_v4","use_source_bbox":True,"bbox_pad":float(ck.get("bbox_pad",0.03))},
    }
    m=ck.get("metrics",{}) or {}
    metrics={
        "official_val_mAP@10":float(m.get("mAP@10",m.get("mAP",0.0))),
        "official_val_Rank-1":float(m.get("Rank-1",0.0)),
        "official_val_Rank-5":float(m.get("Rank-5",0.0)),
        "official_val_mAP_full":float(m.get("mAP_full",0.0)),
        "official_val_mINP":float(m.get("mINP",0.0)),
        "v5_native_metrics":copy.deepcopy(m),
    }
    payload={
        "model":map_v5_state_to_project(ck["model_state"]),
        "model_cfg":model_cfg,
        "label_map":copy.deepcopy(ck.get("class_map",{})),
        "epoch":int(ck.get("epoch",0)),
        "metrics":metrics,
        "source_recipe":f"vehicle_reid_v5_official/train_best_v5.py:{'base' if profile in ('convnext_base','vit_base') else ('large' if profile in ('convnext_large','vit_large') else profile)}",
        "v5_exact_source_checkpoint":str(src),
        "v5_exact_vendor_sha256":"1d8a2872f5f6e46cbabee3f67183ebd1f25437252335752033c7a6b21b83fc4c",
        "preprocess":{"image_size":[h,w],"augmentation_profile":"baseline_v4","use_source_bbox":True,"bbox_pad":float(ck.get("bbox_pad",0.03))},
    }
    torch.save(payload,dst)
    return dst
