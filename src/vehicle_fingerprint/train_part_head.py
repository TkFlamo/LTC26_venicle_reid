from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.part_supervision import PartBootstrapDataset
from .losses import weak_multilabel_part_loss, target_part_visibility
from .train_reid import _load_state_flexible, _model_from_cfg
from .data.schema import PART_SLOT_IDS
from .utils import autocast_context, load_yaml, resolve_device, save_json, seed_everything


def _collate(batch):
    out={}
    for k in batch[0]: out[k]=[x[k] for x in batch] if k=="path" else torch.stack([x[k] for x in batch])
    return out


def _seg_metrics(logits,target,class_sup,threshold=0.5):
    if target.shape[-2:]!=logits.shape[-2:]: target=torch.nn.functional.interpolate(target.float(),size=logits.shape[-2:],mode="area")
    gt=target>0.25; pred=torch.sigmoid(logits.float())>=threshold; sup=class_sup.bool()[:,:,None,None]
    inter=(pred&gt&sup).sum((0,2,3)).float(); union=((pred|gt)&sup).sum((0,2,3)).float(); denom=((pred&sup).sum((0,2,3))+(gt&sup).sum((0,2,3))).float()
    iou=torch.where(union>0,inter/union.clamp_min(1),torch.nan); f1=torch.where(denom>0,2*inter/denom.clamp_min(1),torch.nan)
    return iou,f1


def train_part_head_from_config(
    config_path: str|Path, *, run_dir: str | None = None, warmstart: str | None = None,
    expect_backbone: str | None = None,
):
    cfg=load_yaml(config_path)
    if run_dir is not None: cfg["run_dir"] = str(run_dir)
    if warmstart is not None: cfg["warmstart"] = str(warmstart)
    seed_everything(int(cfg.get("seed",42))); device=resolve_device(cfg.get("device","auto")); precision=cfg.get("precision","bf16")
    run=Path(cfg["run_dir"]);run.mkdir(parents=True,exist_ok=True);save_json(cfg,run/"config.json")
    ck=torch.load(cfg["warmstart"],map_location="cpu",weights_only=False);mcfg=copy.deepcopy(ck["model_cfg"]);mcfg["backbone"]["pretrained"]=False;mcfg["backbone"]["checkpoint_path"]=None
    if expect_backbone is not None:
        from .models.backbone_registry import canonical_backbone_name, profile_config
        expected = profile_config(expect_backbone)["model_name"]
        actual = str(mcfg.get("backbone",{}).get("model_name",""))
        if actual != expected:
            raise RuntimeError(f"Part-head warmstart backbone mismatch: expected {expected!r}, checkpoint has {actual!r}")
    for k,v in cfg.get("model_overrides",{}).items():mcfg[k]=v
    mcfg["enable_parts"]=True;model=_model_from_cfg(mcfg,0).to(device);missing,unexpected,skipped=_load_state_flexible(model,{k:v for k,v in ck["model"].items() if not k.startswith("classifier.")});print(f"Warmstart: missing={len(missing)} unexpected={len(unexpected)} skipped={len(skipped)}")
    for p in model.parameters():p.requires_grad_(False)
    for p in model.part_head.parameters():p.requires_grad_(True)
    for p in model.part_attention.parameters():p.requires_grad_(True)
    if model.spatial_fusion is not None:
        for p in model.spatial_fusion.parameters(): p.requires_grad_(True)
    model.global_bn.eval();model.backbone.eval()

    d=cfg.get("data",{});root=cfg["part_dataset"]
    tr=PartBootstrapDataset(root,"train",image_size=d.get("image_size",[256,384]),target_size=d.get("part_target_size",[64,96]),train=True)
    va=PartBootstrapDataset(root,"val",image_size=d.get("image_size",[256,384]),target_size=d.get("part_target_size",[64,96]),train=False)
    if not len(tr):raise RuntimeError(f"No Carparts-Seg samples under {root}")
    print(f"Carparts bootstrap: train={len(tr)} val={len(va)}")
    tl=DataLoader(tr,batch_size=d.get("batch_size",48),shuffle=True,num_workers=d.get("workers",8),pin_memory=True,persistent_workers=d.get("workers",8)>0,collate_fn=_collate)
    vl=DataLoader(va,batch_size=d.get("eval_batch_size",64),shuffle=False,num_workers=d.get("workers",8),pin_memory=True,collate_fn=_collate) if len(va) else None
    oc=cfg.get("optimizer",{});trainable=list(model.part_head.parameters())+list(model.part_attention.parameters())+(list(model.spatial_fusion.parameters()) if model.spatial_fusion is not None else []);opt=torch.optim.AdamW(trainable,lr=float(oc.get("lr",3e-4)),weight_decay=float(oc.get("weight_decay",.02)));epochs=int(cfg.get("epochs",18));sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs),eta_min=float(oc.get("min_lr",2e-5)));scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and str(precision).lower() in {"fp16","float16"});lcfg=cfg.get("loss",{});best=-1.;history=[]
    for epoch in range(1,epochs+1):
        model.train();model.backbone.eval();model.global_bn.eval();losses=[]
        for b in tqdm(tl,desc=f"carparts {epoch}/{epochs}"):
            image=b["image"].to(device,non_blocking=True);target=b["part_target"].to(device,non_blocking=True);sup=b["part_class_supervision"].to(device,non_blocking=True);avail=b["part_available"].to(device);quality=b["part_quality"].to(device);negw=b["part_negative_weight"].to(device)
            with autocast_context(device,precision):
                with torch.no_grad(): bo=model.backbone(image)
                dense=model.spatial_features(bo)
                logits=model.part_head(dense);seg,diag=weak_multilabel_part_loss(logits,target,sup,avail,quality,negw,focal_gamma=float(lcfg.get("focal_gamma",1.5)),dice_weight=float(lcfg.get("dice_weight",.5)))
                probs=torch.sigmoid(logits);pred,_,_=model.part_attention(dense,probs[:,PART_SLOT_IDS]);oracle=model.part_attention.pool_with_mask(dense,target[:,PART_SLOT_IDS].float()).detach();vis=target_part_visibility(target.float(),PART_SLOT_IDS);slot_sup=sup[:,PART_SLOT_IDS].bool()&vis
                if slot_sup.any():
                    align=1-torch.nn.functional.cosine_similarity(pred.float(),oracle.float(),dim=-1);align_loss=align[slot_sup].mean()
                else:align_loss=pred.sum()*0
                loss=seg+float(lcfg.get("query_alignment",.35))*align_loss
            opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(trainable,float(cfg.get("grad_clip",1.0)));scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
        sch.step();metrics={"epoch":epoch,"train_loss":float(np.mean(losses))};score=-metrics["train_loss"]
        if vl:
            model.eval();ious=[];f1s=[]
            with torch.inference_mode():
                for b in tqdm(vl,desc="part val",leave=False):
                    image=b["image"].to(device);target=b["part_target"].to(device);sup=b["part_class_supervision"].to(device)
                    with autocast_context(device,precision):
                        bo=model.backbone(image); dense=model.spatial_features(bo); logits=model.part_head(dense)
                    i,f=_seg_metrics(logits,target,sup);ious.append(i.cpu());f1s.append(f.cpu())
            i=torch.stack(ious).nanmean(0);f=torch.stack(f1s).nanmean(0);metrics["val_macro_iou"]=float(torch.nanmean(i));metrics["val_macro_f1"]=float(torch.nanmean(f));score=metrics["val_macro_f1"]
        history.append(metrics);print(metrics);save_json(history,run/"history.json")
        savecfg=copy.deepcopy(mcfg);savecfg["backbone"]["pretrained"]=False;savecfg["backbone"]["checkpoint_path"]=None;savecfg["preprocess"]={"image_size":d.get("image_size",[256,384]),"augmentation_profile":"baseline_v4","use_source_bbox":True,"bbox_pad":0.03};payload={"model":model.state_dict(),"model_cfg":savecfg,"epoch":epoch,"metrics":metrics};torch.save(payload,run/"last.pt")
        if score>best:best=score;torch.save(payload,run/"best.pt")
    return run/"best.pt"
