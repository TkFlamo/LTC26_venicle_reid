from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import ReIDDataset
from .data.sampler import PKBatchSampler
from .losses import (
    batch_hard_triplet,
    supervised_contrastive,
    part_consistency,
    part_supervised_contrastive,
    part_batch_hard_triplet,
    cosine_distillation,
    part_probability_distillation, part_presence_distillation,
    encode_camera_ids, cross_camera_soft_batch_hard_triplet,
    cross_camera_supervised_contrastive, CrossBatchMemory, cross_batch_memory_triplet,
)
from .metrics import cross_camera_retrieval_metrics
from .official_eval import (
    blended_embedding, flatten_official_report, official_metrics_from_embeddings,
    official_selection_key, protocol_ids,
)
from .models.reid import VehicleFingerprintModel
from .utils import load_yaml, save_json, seed_everything, resolve_device, autocast_context
from .models.backbone_registry import apply_backbone_override
from .hard_mining import mine_identity_hard_negatives_from_arrays

META_KEYS = {"vehicle_key", "camera_id", "sample_id", "path", "orig_size"}


def _collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = [x[k] for x in batch] if k in META_KEYS else torch.stack([x[k] for x in batch])
    return out


def build_label_map(df: pd.DataFrame) -> dict[str, int]:
    if "dataset" not in df.columns:
        df = df.copy(); df["dataset"] = "hackathon"
    keys = sorted(set(df.dataset.astype(str) + ":" + df.vehicle_id.astype(str)))
    return {k: i for i, k in enumerate(keys)}


def _load_state_flexible(model: torch.nn.Module, state: dict[str, torch.Tensor]):
    current = model.state_dict()
    keep = {k: v for k, v in state.items() if k in current and tuple(current[k].shape) == tuple(v.shape)}
    skipped = sorted(set(state) - set(keep))
    missing, unexpected = model.load_state_dict(keep, strict=False)
    return missing, unexpected, skipped


def _model_from_cfg(model_cfg: dict, num_classes: int) -> VehicleFingerprintModel:
    return VehicleFingerprintModel(
        model_cfg["backbone"], num_classes=num_classes,
        embed_dim=model_cfg.get("embed_dim", 512),
        part_dim=model_cfg.get("part_dim", 128),
        local_dim=model_cfg.get("local_dim", 128),
        local_grid=tuple(model_cfg.get("local_grid", [4, 6])),
        fusion_layers=model_cfg.get("fusion_layers", 2),
        fusion_heads=model_cfg.get("fusion_heads", 8),
        arc_scale=model_cfg.get("arc_scale", 30.0),
        arc_margin=model_cfg.get("arc_margin", 0.35),
        part_head_dim=model_cfg.get("part_head_dim", 256),
        part_head_blocks=model_cfg.get("part_head_blocks", 2),
        part_attention_heads=model_cfg.get("part_attention_heads", 8),
        part_prior_strength=model_cfg.get("part_prior_strength", 2.5),
        part_visibility_threshold=model_cfg.get("part_visibility_threshold", 0.35),
        part_visibility_topk=model_cfg.get("part_visibility_topk", 3),
        detach_semantic_prior=model_cfg.get("detach_semantic_prior", True),
        part_dropout=model_cfg.get("part_dropout", 0.08),
        fusion_max_residual=model_cfg.get("fusion_max_residual", 0.25),
        fusion_initial_gate=model_cfg.get("fusion_initial_gate", 0.01),
        enable_parts=model_cfg.get("enable_parts", True),
        global_head_mode=model_cfg.get("global_head_mode", "baseline_v4"),
        multiscale_spatial=model_cfg.get("multiscale_spatial", False),
        spatial_fusion_dim=model_cfg.get("spatial_fusion_dim", 256),
    )


def _extract(model, loader, device, precision):
    model.eval(); zg=[]; zf=[]; zl=[]; vids=[]; cams=[]
    with torch.inference_mode():
        for b in tqdm(loader, desc="val", leave=False):
            image = b["image"].to(device, non_blocking=True)
            with autocast_context(device, precision):
                o = model(image)
            zg.append(o["z_global"].float().cpu().numpy())
            zf.append(o["z_fused"].float().cpu().numpy())
            zl.append(o["z_local"].float().cpu().numpy())
            vids.extend(b["vehicle_key"]); cams.extend(b["camera_id"])
    return np.concatenate(zg), np.concatenate(zf), np.concatenate(zl), np.asarray(vids), np.asarray(cams)


def _blend_embeddings(zg: np.ndarray, zf: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    zg = zg.astype(np.float32, copy=True); zf = zf.astype(np.float32, copy=True)
    zg /= np.linalg.norm(zg, axis=1, keepdims=True).clip(1e-12)
    zf /= np.linalg.norm(zf, axis=1, keepdims=True).clip(1e-12)
    return np.concatenate([np.sqrt(1.0-a)*zg, np.sqrt(a)*zf], axis=1)


def _evaluate_branches(zg, zf, vids, cams, same_camera_policy: str, alpha_grid: list[float]):
    # Legacy diagnostic only. Official model selection uses the organizers' fixed query/gallery protocol below.
    global_m = cross_camera_retrieval_metrics(zg, vids, cams, same_camera_policy=same_camera_policy)
    fused_m = cross_camera_retrieval_metrics(zf, vids, cams, same_camera_policy=same_camera_policy)
    best = None
    for alpha in alpha_grid:
        m = cross_camera_retrieval_metrics(_blend_embeddings(zg, zf, alpha), vids, cams, same_camera_policy=same_camera_policy)
        key = (m["mAP"], m.get("Rank-1",0.0), m.get("Rank-5",0.0))
        if best is None or key > best[0]: best = (key,float(alpha),m)
    return global_m, fused_m, best[1], best[2]


def _evaluate_official_branches(qzg, qzf, gzg, gzf, q_ids, g_ids, gt_csv, alpha_grid, evaluator_path=None, top_k=10):
    rg = official_metrics_from_embeddings(qzg, gzg, q_ids=q_ids, g_ids=g_ids, gt_csv=gt_csv, top_k=top_k, evaluator_path=evaluator_path)
    rf = official_metrics_from_embeddings(qzf, gzf, q_ids=q_ids, g_ids=g_ids, gt_csv=gt_csv, top_k=top_k, evaluator_path=evaluator_path)
    best = None
    for alpha in alpha_grid:
        qe = blended_embedding(qzg, qzf, alpha); ge = blended_embedding(gzg, gzf, alpha)
        r = official_metrics_from_embeddings(qe, ge, q_ids=q_ids, g_ids=g_ids, gt_csv=gt_csv, top_k=top_k, evaluator_path=evaluator_path)
        key = official_selection_key(r)
        if best is None or key > best[0]:
            best = (key, float(alpha), r)
    return rg, rf, best[1], best[2], best[0]


def _set_trainability(model: VehicleFingerprintModel, cfg: dict, arcface_weight: float):
    ocfg = cfg.get("optimizer", {})
    # Start from trainable and apply explicit freezes. This makes stage configs auditable.
    for p in model.parameters(): p.requires_grad_(True)
    freeze_backbone = bool(ocfg.get("freeze_backbone", False))
    last_stages = int(ocfg.get("unfreeze_backbone_last_units", ocfg.get("unfreeze_backbone_last_stages", 0)))
    if freeze_backbone or last_stages > 0:
        model.backbone.freeze_all()
    if last_stages > 0:
        model.backbone.unfreeze_last_units(last_stages)
    freeze_global = bool(ocfg.get("freeze_global_head", False))
    if freeze_global:
        for p in model.global_proj.parameters(): p.requires_grad_(False)
        for p in model.global_bn.parameters(): p.requires_grad_(False)
    if bool(ocfg.get("freeze_part_head", False)):
        for p in model.part_head.parameters(): p.requires_grad_(False)
    if bool(ocfg.get("freeze_part_attention", False)):
        for p in model.part_attention.parameters(): p.requires_grad_(False)
    if model.classifier is not None and arcface_weight <= 0:
        for p in model.classifier.parameters(): p.requires_grad_(False)
    return {
        "freeze_backbone": freeze_backbone,
        "unfreeze_backbone_last_units": last_stages,
        "freeze_global_head": freeze_global,
        "freeze_part_head": bool(ocfg.get("freeze_part_head", False)),
    }


def _load_teacher(path: str | None, device):
    if not path: return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = copy.deepcopy(ck["model_cfg"]); mcfg["backbone"]["pretrained"] = False; mcfg["backbone"]["checkpoint_path"] = None
    teacher = _model_from_cfg(mcfg, 0).to(device)
    state = {k:v for k,v in ck["model"].items() if not k.startswith("classifier.")}
    _load_state_flexible(teacher, state)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    return teacher


def train_from_config(
    config_path: str | Path, *, backbone: str | None = None, run_dir: str | None = None,
    warmstart: str | None = None, teacher_checkpoint: str | None = None,
    hard_negative_map: str | None = None,
):
    cfg = load_yaml(config_path)
    cfg = apply_backbone_override(cfg, backbone)
    if run_dir is not None: cfg["run_dir"] = str(run_dir)
    if warmstart is not None: cfg["warmstart"] = None if str(warmstart).lower() in {"none","null",""} else str(warmstart)
    if teacher_checkpoint is not None: cfg["teacher_checkpoint"] = None if str(teacher_checkpoint).lower() in {"none","null",""} else str(teacher_checkpoint)
    if hard_negative_map is not None: cfg["hard_negative_map"] = None if str(hard_negative_map).lower() in {"none","null",""} else str(hard_negative_map)
    seed_everything(int(cfg.get("seed",42)))
    device = resolve_device(cfg.get("device","auto")); precision = cfg.get("precision","bf16")
    run_dir = Path(cfg["run_dir"]); run_dir.mkdir(parents=True,exist_ok=True); save_json(cfg,run_dir/"config.json")
    train_df = pd.read_csv(cfg["train_manifest"]); val_df = pd.read_csv(cfg["val_manifest"]) if cfg.get("val_manifest") else None
    if "dataset" not in train_df.columns: train_df["dataset"]="hackathon"
    if val_df is not None and "dataset" not in val_df.columns: val_df["dataset"]="hackathon"
    label_map = build_label_map(train_df); (run_dir/"label_map.json").write_text(json.dumps(label_map,ensure_ascii=False,indent=2),encoding="utf-8")

    dcfg = cfg.get("data",{})
    ds_kwargs = dict(
        image_size=dcfg.get("image_size",[256,384]),
        augmentation_profile=dcfg.get("augmentation_profile","baseline_v4"),
        use_source_bbox=dcfg.get("use_source_bbox",True),
        bbox_pad=dcfg.get("bbox_pad",0.03),
        part_target_size=dcfg.get("part_target_size",[64,96]),
        return_weak_view=bool(dcfg.get("return_weak_view",False)),
    )
    train_ds = ReIDDataset(train_df,train=True,label_map=label_map,**ds_kwargs)
    sampler = PKBatchSampler(train_ds,p=dcfg.get("p",16),k=dcfg.get("k",4),batches_per_epoch=dcfg.get("batches_per_epoch"),hard_map=cfg.get("hard_negative_map"),seed=cfg.get("seed",42))
    loader = DataLoader(train_ds,batch_sampler=sampler,num_workers=dcfg.get("workers",8),pin_memory=True,persistent_workers=dcfg.get("workers",8)>0,collate_fn=_collate)
    online_every=int(cfg.get("online_hard_mining_every",0)); mine_loader=None
    if online_every>0:
        mine_ds=ReIDDataset(train_df,train=False,label_map=label_map,**ds_kwargs)
        mine_loader=DataLoader(mine_ds,batch_size=dcfg.get("eval_batch_size",64),shuffle=False,num_workers=dcfg.get("workers",8),pin_memory=True,collate_fn=_collate)

    weights = cfg.get("loss",{})
    arc_w=float(weights.get("arcface",0)); gtri_w=float(weights.get("global_triplet",0)); ftri_w=float(weights.get("fused_triplet",0)); sup_w=float(weights.get("global_supcon",0)); ltri_w=float(weights.get("local_triplet",0))
    pcons_w=float(weights.get("part_consistency",0)); psup_w=float(weights.get("part_supcon",0)); ptri_w=float(weights.get("part_triplet",0))
    gdist_w=float(weights.get("global_distill",0)); pdist_w=float(weights.get("part_prob_distill",0)); ppres_w=float(weights.get("ema_part_presence",0.0))
    ema_decay=float(weights.get("ema_teacher_decay",0.0)); ema_topk=int(weights.get("ema_presence_topk",4))
    use_cross=bool(weights.get("cross_camera_triplet",True)); cross_ratio=float(weights.get("cross_camera_positive_weight",0.75)); any_ratio=float(weights.get("same_camera_positive_weight",0.25))
    memory_w=float(weights.get("memory_triplet_weight",0.0)); memory=CrossBatchMemory(int(weights.get("memory_bank_size",0)))

    model_cfg=copy.deepcopy(cfg["model"])
    if cfg.get("warmstart"):
        model_cfg["backbone"]["pretrained"]=False; model_cfg["backbone"]["checkpoint_path"]=None
    model=_model_from_cfg(model_cfg,len(label_map)).to(device)
    if cfg.get("warmstart"):
        ck=torch.load(cfg["warmstart"],map_location="cpu",weights_only=False)
        missing,unexpected,skipped=_load_state_flexible(model,ck.get("model",ck))
        print(f"Warmstart: missing={len(missing)} unexpected={len(unexpected)} skipped={len(skipped)}")
        if skipped: print("Warmstart skipped examples:",skipped[:8])
    teacher=_load_teacher(cfg.get("teacher_checkpoint"),device)
    # Keep the explicit teacher checkpoint fixed for global/Carparts preservation.  Target-domain
    # semantic self-training uses a separate EMA copy of the current student; otherwise updating the
    # same teacher would silently weaken global distillation over time.
    ema_teacher=None
    if ppres_w>0 and ema_decay>0:
        ema_teacher=copy.deepcopy(model).to(device).eval()
        for p in ema_teacher.parameters(): p.requires_grad_(False)
    freeze_info=_set_trainability(model,cfg,arc_w); print("Trainability:",freeze_info)

    ocfg=cfg.get("optimizer",{}); wd=float(ocfg.get("weight_decay",0.03)); groups=[]
    buckets={"backbone":[],"parts":[],"heads":[]}
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("backbone."): buckets["backbone"].append(p)
        elif n.startswith("part_head.") or n.startswith("part_attention.") or n.startswith("spatial_fusion."): buckets["parts"].append(p)
        else: buckets["heads"].append(p)
    lrs={"backbone":float(ocfg.get("backbone_lr",2e-6)),"parts":float(ocfg.get("part_head_lr",8e-5)),"heads":float(ocfg.get("head_lr",1e-4))}
    for name,params in buckets.items():
        if params: groups.append({"params":params,"lr":lrs[name],"name":name})
    if not groups: raise RuntimeError("No trainable parameters")
    optimizer=torch.optim.AdamW(groups,weight_decay=wd)
    epochs=int(cfg.get("epochs",20)); steps=max(1,epochs*len(loader)); warm=max(1,int(steps*float(ocfg.get("warmup_fraction",0.03))))
    def lr_lambda(step):
        if step<warm:return max(1e-3,(step+1)/warm)
        t=(step-warm)/max(1,steps-warm); return 0.5*(1+np.cos(np.pi*t))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and str(precision).lower() in {"fp16","float16"})

    tri_margin=float(weights.get("triplet_margin",0.20)); local_margin=float(weights.get("local_triplet_margin",0.18)); part_margin=float(weights.get("part_triplet_margin",0.14)); part_temp=float(weights.get("part_temperature",0.08)); min_part_score=float(weights.get("part_metric_min_score",0.42))
    grad_accum=int(cfg.get("grad_accum",1)); clip=float(cfg.get("grad_clip",3.0))

    val_loader=None
    if val_df is not None:
        val_map=build_label_map(val_df); val_ds=ReIDDataset(val_df,train=False,label_map=val_map,**ds_kwargs)
        val_loader=DataLoader(val_ds,batch_size=dcfg.get("eval_batch_size",64),shuffle=False,num_workers=dcfg.get("workers",8),pin_memory=True,collate_fn=_collate)
    ecfg=cfg.get("evaluation",{}); policy=str(ecfg.get("same_camera_policy","same_identity")); alpha_grid=[float(x) for x in ecfg.get("blend_alpha_grid",[0,.05,.1,.2,.35,.5,1])]

    official_cfg=ecfg.get("official",{}) or {}
    official_enabled=bool(official_cfg.get("gt") and official_cfg.get("query_manifest") and official_cfg.get("gallery_manifest"))
    official_q_loader=official_g_loader=None; official_q_ids=official_g_ids=None
    if official_enabled:
        oq=pd.read_csv(official_cfg["query_manifest"]); og=pd.read_csv(official_cfg["gallery_manifest"])
        for odf in (oq,og):
            if "dataset" not in odf.columns: odf["dataset"]="hackathon"
        oq_map=build_label_map(oq); og_map=build_label_map(og)
        oq_ds=ReIDDataset(oq,train=False,label_map=oq_map,**ds_kwargs); og_ds=ReIDDataset(og,train=False,label_map=og_map,**ds_kwargs)
        official_q_loader=DataLoader(oq_ds,batch_size=dcfg.get("eval_batch_size",64),shuffle=False,num_workers=dcfg.get("workers",8),pin_memory=True,collate_fn=_collate)
        official_g_loader=DataLoader(og_ds,batch_size=dcfg.get("eval_batch_size",64),shuffle=False,num_workers=dcfg.get("workers",8),pin_memory=True,collate_fn=_collate)
        official_q_ids,official_g_ids=protocol_ids(official_cfg["query_manifest"],official_cfg["gallery_manifest"])

    best_key=None; history=[]
    for epoch in range(1,epochs+1):
        model.train()
        if freeze_info["freeze_backbone"]:
            model.backbone.eval()
        elif freeze_info["unfreeze_backbone_last_units"] > 0:
            # Keep frozen early backbone units deterministic.  For ConvNeXt the units are stages;
            # for ViT they are transformer blocks.
            model.backbone.set_last_units_train(freeze_info["unfreeze_backbone_last_units"])
        if freeze_info["freeze_global_head"]: model.global_bn.eval()
        optimizer.zero_grad(set_to_none=True); run=[]; comp=[]
        pbar=tqdm(loader,desc=f"detail epoch {epoch}/{epochs}")
        for it,b in enumerate(pbar,1):
            image=b["image"].to(device,non_blocking=True); labels=b["label"].to(device,non_blocking=True); cams_t=encode_camera_ids(b["camera_id"],device)
            with autocast_context(device,precision):
                o=model(image,labels if arc_w>0 else None)
                zero=o["z_global"].sum()*0
                l_arc=F.cross_entropy(o["logits"],labels,label_smoothing=float(weights.get("label_smoothing",0.0))) if arc_w>0 and o["logits"] is not None else zero
                l_gtri=(cross_camera_soft_batch_hard_triplet(o["z_global"],labels,cams_t,margin=tri_margin,cross_camera_weight=cross_ratio,any_camera_weight=any_ratio) if use_cross else batch_hard_triplet(o["z_global"],labels,margin=tri_margin)) if gtri_w>0 else zero
                l_ftri=(cross_camera_soft_batch_hard_triplet(o["z_fused"],labels,cams_t,margin=tri_margin,cross_camera_weight=cross_ratio,any_camera_weight=any_ratio) if use_cross else batch_hard_triplet(o["z_fused"],labels,margin=tri_margin)) if ftri_w>0 else zero
                l_sup=(cross_camera_supervised_contrastive(o["z_global"],labels,cams_t,cross_camera_positive_weight=float(weights.get("supcon_cross_camera_weight",2.0)),same_camera_positive_weight=float(weights.get("supcon_same_camera_weight",0.5))) if use_cross else supervised_contrastive(o["z_global"],labels)) if sup_w>0 else zero
                l_ltri=(cross_camera_soft_batch_hard_triplet(o["z_local"],labels,cams_t,margin=local_margin,cross_camera_weight=cross_ratio,any_camera_weight=any_ratio) if use_cross else batch_hard_triplet(o["z_local"],labels,margin=local_margin)) if ltri_w>0 else zero
                l_mem=cross_batch_memory_triplet(o["z_global"],labels,cams_t,memory,margin=tri_margin) if memory_w>0 else zero
                vis=o["visibility"] & (o["visibility_score"]>=min_part_score)
                l_pc=part_consistency(o["parts"],vis,labels) if pcons_w>0 else zero
                l_ps=part_supervised_contrastive(o["parts"],vis,labels,temperature=part_temp) if psup_w>0 else zero
                l_pt=part_batch_hard_triplet(o["parts"],vis,labels,margin=part_margin) if ptri_w>0 else zero
                l_gd=l_pd=l_pp=zero
                if (teacher is not None and (gdist_w>0 or pdist_w>0)) or (ema_teacher is not None and ppres_w>0):
                    with torch.no_grad():
                        t=teacher(image) if teacher is not None and (gdist_w>0 or pdist_w>0) else None
                        tw=ema_teacher(b["weak_image"].to(device,non_blocking=True)) if ema_teacher is not None and ppres_w>0 and "weak_image" in b else None
                    if gdist_w>0 and t is not None: l_gd=cosine_distillation(o["z_global"],t["z_global"])
                    if pdist_w>0 and t is not None and t["part_logits"].shape==o["part_logits"].shape: l_pd=part_probability_distillation(o["part_logits"],t["part_logits"])
                    if ppres_w>0 and tw is not None: l_pp=part_presence_distillation(o["part_logits"],tw["part_logits"],topk=ema_topk)
                loss=(arc_w*l_arc+gtri_w*l_gtri+ftri_w*l_ftri+sup_w*l_sup+ltri_w*l_ltri+memory_w*l_mem+pcons_w*l_pc+psup_w*l_ps+ptri_w*l_pt+gdist_w*l_gd+pdist_w*l_pd+ppres_w*l_pp)/grad_accum
            scaler.scale(loss).backward()
            if it%grad_accum==0:
                scaler.unscale_(optimizer); trainable=[p for p in model.parameters() if p.requires_grad and p.grad is not None]
                if clip>0: torch.nn.utils.clip_grad_norm_(trainable,clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step(); memory.update(o["z_global"],labels,cams_t)
                if ema_teacher is not None and ema_decay>0:
                    with torch.no_grad():
                        sp=dict(model.named_parameters())
                        for n,tp in ema_teacher.named_parameters():
                            if n in sp and tp.shape==sp[n].shape: tp.mul_(ema_decay).add_(sp[n].detach(),alpha=1.0-ema_decay)
                        # BN buffers are cheap to track directly; otherwise EMA teacher statistics
                        # stay stuck at initialization while its weights evolve.
                        sb=dict(model.named_buffers())
                        for n,tb in ema_teacher.named_buffers():
                            if n in sb and tb.shape==sb[n].shape: tb.copy_(sb[n].detach())
            run.append(float(loss.detach().cpu())*grad_accum)
            comp.append({"arc":float(l_arc.detach()),"gtri":float(l_gtri.detach()),"ftri":float(l_ftri.detach()),"local":float(l_ltri.detach()),"memory":float(l_mem.detach()),"pcons":float(l_pc.detach()),"psup":float(l_ps.detach()),"ptri":float(l_pt.detach()),"gdist":float(l_gd.detach()),"pdist":float(l_pd.detach()),"ppres":float(l_pp.detach()),"vis":float(vis.float().mean().detach())})
            pbar.set_postfix(loss=f"{np.mean(run[-50:]):.4f}",vis=f"{comp[-1]['vis']:.2f}")

        metrics={"epoch":epoch,"train_loss":float(np.mean(run)),"fusion_gate":float(model.fusion.gate.detach().cpu())}
        for k in comp[0]: metrics[f"train_{k}"]=float(np.mean([x[k] for x in comp]))
        do_eval=(epoch%int(cfg.get("eval_every",1))==0 or epoch==epochs)
        if official_enabled and do_eval:
            qzg,qzf,_qzl,_qv,_qc=_extract(model,official_q_loader,device,precision)
            gzg,gzf,_gzl,_gv,_gc=_extract(model,official_g_loader,device,precision)
            rg,rf,alpha,rb,key=_evaluate_official_branches(
                qzg,qzf,gzg,gzf,official_q_ids,official_g_ids,official_cfg["gt"],alpha_grid,
                evaluator_path=official_cfg.get("evaluator"),top_k=int(official_cfg.get("top_k",10)),
            )
            metrics.update(flatten_official_report(rg,"official_val_global"))
            metrics.update(flatten_official_report(rf,"official_val_fused"))
            metrics.update(flatten_official_report(rb,"official_val_best"))
            metrics["official_val_best_alpha"]=alpha
            # Optional legacy all-vs-all diagnostics are logged but never select the checkpoint.
            if val_loader is not None:
                zg,zf,_zl,vids,cams=_extract(model,val_loader,device,precision); mg,mf,la,lm=_evaluate_branches(zg,zf,vids,cams,policy,alpha_grid)
                metrics["legacy_val_global_mAP"]=mg["mAP"]; metrics["legacy_val_fused_mAP"]=mf["mAP"]; metrics["legacy_val_best_mAP"]=lm["mAP"]; metrics["legacy_val_best_alpha"]=la
        elif val_loader is not None and do_eval:
            zg,zf,_zl,vids,cams=_extract(model,val_loader,device,precision); mg,mf,alpha,mb=_evaluate_branches(zg,zf,vids,cams,policy,alpha_grid)
            for k,v in mg.items(): metrics[f"val_global_{k}"]=v
            for k,v in mf.items(): metrics[f"val_fused_{k}"]=v
            for k,v in mb.items(): metrics[f"val_best_{k}"]=v
            metrics["val_best_alpha"]=alpha; key=(mb["mAP"],mb.get("Rank-1",0),mb.get("Rank-5",0),0.,0.)
        else:
            key=(-metrics["train_loss"],0.,0.,0.,0.)
        if mine_loader is not None and online_every>0 and epoch%online_every==0:
            mzg,_mzf,_mzl,mvids,_mcams=_extract(model,mine_loader,device,precision)
            hmap=mine_identity_hard_negatives_from_arrays(mzg,mvids,topk=int(cfg.get("online_hard_topk",40)),refine_factor=int(cfg.get("online_hard_refine_factor",4)),top_pair_mean=int(cfg.get("online_hard_top_pair_mean",3)))
            sampler.set_hard_map(hmap)
            (run_dir/f"hard_negatives_epoch_{epoch:03d}.json").write_text(json.dumps(hmap,ensure_ascii=False,indent=2),encoding="utf-8")
            metrics["online_hard_map_ids"]=len(hmap)
        history.append(metrics); print(json.dumps(metrics,ensure_ascii=False)); save_json(history,run_dir/"history.json")
        ckcfg=copy.deepcopy(model_cfg); ckcfg["backbone"]["pretrained"]=False; ckcfg["backbone"]["checkpoint_path"]=None
        ckcfg["preprocess"]={"image_size":dcfg.get("image_size",[256,384]),"augmentation_profile":dcfg.get("augmentation_profile","baseline_v4"),"use_source_bbox":dcfg.get("use_source_bbox",True),"bbox_pad":dcfg.get("bbox_pad",0.03)}
        payload={"model":model.state_dict(),"model_cfg":ckcfg,"label_map":label_map,"epoch":epoch,"metrics":metrics}
        torch.save(payload,run_dir/"last.pt")
        if best_key is None or key>best_key:
            best_key=key; payload["official_selection_key"]=list(key); torch.save(payload,run_dir/"best.pt")
    return run_dir/"best.pt"
