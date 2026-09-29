from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from .backbone_registry import resolve_backbone_kwargs, infer_family, profile_config


@dataclass
class DenseBackboneOutput:
    global_feat: torch.Tensor
    feature_map: torch.Tensor
    feature_maps: list[torch.Tensor] | None = None


class DinoV3Backbone(nn.Module):
    """DINOv3 backbone exposing both a strong global descriptor source and spatial pyramids.

    The global path is deliberately kept baseline-compatible.  Spatial taps are independent:
    ConvNeXt can expose several stages and ViT several transformer depths.  A lightweight fusion
    module in :class:`VehicleFingerprintModel` decides how to combine them, so Carparts/detail
    training can learn the spatial branch without changing the pretrained global path.
    """
    def __init__(
        self,
        model_name: str = "convnext_small.dinov3_lvd1689m",
        *,
        profile: str | None = None,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
        cache_dir: str | None = None,
        img_size: int | tuple[int, int] | None = None,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = True,
        dense_stage: int | None = -2,
        dense_stages: Iterable[int] | None = None,
        vit_dense_blocks: Iterable[int] | None = None,
        global_feature_mode: str = "timm_prelogits",
    ):
        super().__init__()
        import timm

        raw_cfg = {
            "pretrained": pretrained, "checkpoint_path": checkpoint_path, "cache_dir": cache_dir,
            "img_size": img_size, "dynamic_img_size": dynamic_img_size,
            "dynamic_img_pad": dynamic_img_pad,
        }
        if profile is not None:
            raw_cfg["profile"] = profile
            prof = profile_config(profile)
            raw_cfg["model_name"] = prof["model_name"] if model_name == "convnext_small.dinov3_lvd1689m" else model_name
            raw_cfg["dense_stage"] = prof.get("dense_stage") if dense_stage == -2 else dense_stage
            raw_cfg["dense_stages"] = list(prof.get("dense_stages") or []) if dense_stages is None else list(dense_stages)
            raw_cfg["vit_dense_blocks"] = list(prof.get("vit_dense_blocks") or []) if vit_dense_blocks is None else list(vit_dense_blocks)
        else:
            raw_cfg["model_name"] = model_name; raw_cfg["dense_stage"] = dense_stage
            if dense_stages is not None: raw_cfg["dense_stages"] = list(dense_stages)
            if vit_dense_blocks is not None: raw_cfg["vit_dense_blocks"] = list(vit_dense_blocks)
        resolved = resolve_backbone_kwargs(raw_cfg)
        self.profile = resolved.get("profile")
        self.global_feature_mode = str(resolved.get("global_feature_mode", global_feature_mode))
        self.model_name = str(resolved.get("model_name", model_name))
        self.family = infer_family(self.model_name)
        pretrained = bool(resolved.get("pretrained", pretrained))
        checkpoint_path = resolved.get("checkpoint_path", checkpoint_path)
        cache_dir = resolved.get("cache_dir", cache_dir)
        img_size = resolved.get("img_size", img_size)
        dynamic_img_size = bool(resolved.get("dynamic_img_size", dynamic_img_size))
        dynamic_img_pad = bool(resolved.get("dynamic_img_pad", dynamic_img_pad))
        dense_stage = resolved.get("dense_stage", dense_stage)
        dense_stages = resolved.get("dense_stages", dense_stages)
        vit_dense_blocks = resolved.get("vit_dense_blocks", vit_dense_blocks)

        kwargs: dict = {"pretrained": pretrained, "num_classes": 0}
        if cache_dir: kwargs["cache_dir"] = str(cache_dir)
        if self.family == "vit":
            if img_size is not None: kwargs["img_size"] = img_size
            kwargs["dynamic_img_size"] = dynamic_img_size; kwargs["dynamic_img_pad"] = dynamic_img_pad
        if checkpoint_path:
            p = Path(checkpoint_path).expanduser().resolve()
            if not p.is_file(): raise FileNotFoundError(f"Local timm DINOv3 checkpoint not found: {p}")
            kwargs["pretrained_cfg_overlay"] = {"file": str(p), "num_classes": 0}; kwargs["pretrained"] = True

        self.model = timm.create_model(self.model_name, **kwargs)
        self.hidden_size = int(getattr(self.model, "num_features", getattr(self.model, "embed_dim", 768)))
        self.is_vit = bool(hasattr(self.model, "patch_embed") and hasattr(self.model, "num_prefix_tokens"))
        if self.is_vit: self.family = "vit"
        patch = getattr(getattr(self.model, "patch_embed", None), "patch_size", 16)
        self.patch_size = tuple(map(int, patch)) if isinstance(patch, (tuple, list)) else (int(patch), int(patch))

        self._conv_dense_cache: dict[int, torch.Tensor] = {}
        self._conv_dense_handles: list = []
        self._vit_dense_cache: dict[int, torch.Tensor] = {}
        self._vit_dense_handles: list = []
        self.dense_stages: list[int] = []
        self.vit_dense_blocks: list[int] = []
        self.dense_channels: list[int] = []

        stages = getattr(self.model, "stages", None)
        if not self.is_vit and stages is not None and len(stages):
            requested = list(dense_stages or ([] if dense_stage is None else [dense_stage]))
            if not requested: requested = [-2]
            fi = getattr(self.model, "feature_info", None)
            try: fi_channels = list(fi.channels()) if fi is not None else []
            except Exception: fi_channels = []
            seen = set()
            for raw in requested:
                idx = int(raw); idx = idx + len(stages) if idx < 0 else idx; idx = max(0, min(idx, len(stages)-1))
                if idx in seen: continue
                seen.add(idx); self.dense_stages.append(idx)
                ch = int(fi_channels[idx]) if idx < len(fi_channels) else None
                if ch is None:
                    for m in reversed(list(stages[idx].modules())):
                        if hasattr(m, "out_channels"): ch = int(m.out_channels); break
                self.dense_channels.append(int(ch or self.hidden_size))
                def _make_hook(i):
                    def _capture(_module, _inputs, output): self._conv_dense_cache[i] = output
                    return _capture
                self._conv_dense_handles.append(stages[idx].register_forward_hook(_make_hook(idx)))

        if self.is_vit:
            blocks = getattr(self.model, "blocks", None)
            requested = list(vit_dense_blocks or [-1])
            if blocks is not None and len(blocks):
                seen=set()
                for raw in requested:
                    idx=int(raw); idx=idx+len(blocks) if idx<0 else idx; idx=max(0,min(idx,len(blocks)-1))
                    if idx in seen: continue
                    seen.add(idx); self.vit_dense_blocks.append(idx); self.dense_channels.append(self.hidden_size)
                    def _make_hook(i):
                        def _hook(_module,_inputs,output):
                            if torch.is_tensor(output): self._vit_dense_cache[i]=output
                            elif isinstance(output,(tuple,list)) and output and torch.is_tensor(output[0]): self._vit_dense_cache[i]=output[0]
                        return _hook
                    self._vit_dense_handles.append(blocks[idx].register_forward_hook(_make_hook(idx)))
            if not self.dense_channels: self.dense_channels=[self.hidden_size]

        if not self.dense_channels: self.dense_channels=[self.hidden_size]
        self.dense_size = int(self.dense_channels[-1])
        self.dense_stage = self.dense_stages[-1] if self.dense_stages else None

    def _as_nchw(self, feat: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feat.ndim != 4: raise RuntimeError(f"Expected 4D dense feature map, got {tuple(feat.shape)}")
        if feat.shape[1] == expected_channels: return feat
        if feat.shape[-1] == expected_channels: return feat.permute(0,3,1,2).contiguous()
        return feat

    def _num_patches(self,x:torch.Tensor)->tuple[int,int,int]:
        _,_,h,w=x.shape; ph=max(1,(h+self.patch_size[0]-1)//self.patch_size[0]); pw=max(1,(w+self.patch_size[1]-1)//self.patch_size[1]); return ph,pw,ph*pw

    def _normalise_intermediate_tokens(self,t:torch.Tensor)->torch.Tensor:
        norm=getattr(self.model,"norm",None); return norm(t) if norm is not None else t

    def _vit_dense(self,tokens:torch.Tensor,x:torch.Tensor)->DenseBackboneOutput:
        ph,pw,n=self._num_patches(x)
        if tokens.ndim!=3 or tokens.shape[1]<n: raise RuntimeError(f"Unexpected ViT token tensor {tuple(tokens.shape)} for {tuple(x.shape[-2:])}")
        maps=[]
        for idx in self.vit_dense_blocks:
            t=self._vit_dense_cache.get(idx)
            if t is None or t.ndim!=3 or t.shape[1]<n: continue
            t=self._normalise_intermediate_tokens(t)[:, -n:, :]
            maps.append(t.transpose(1,2).reshape(tokens.shape[0],tokens.shape[-1],ph,pw))
        if not maps:
            t=tokens[:,-n:,:]; maps=[t.transpose(1,2).reshape(tokens.shape[0],tokens.shape[-1],ph,pw)]
        fmap=maps[-1]
        if self.global_feature_mode == "v5_avg":
            # vehicle_reid_v5_official with --pooling avg explicitly averages patch
            # tokens for ViT.  Preserve that exact global representation after
            # converting a V5 checkpoint into the spatial/part-aware project.
            global_feat=tokens[:,-n:,:].mean(1)
        elif hasattr(self.model,"forward_head"):
            global_feat=self.model.forward_head(tokens,pre_logits=True)
            if global_feat.ndim!=2: global_feat=tokens[:,-n:,:].mean(1)
        else: global_feat=tokens[:,-n:,:].mean(1)
        return DenseBackboneOutput(global_feat,fmap,maps)

    def backbone_units(self)->list[nn.Module]:
        if self.is_vit:
            b=getattr(self.model,"blocks",None); return list(b) if b is not None else []
        s=getattr(self.model,"stages",None); return list(s) if s is not None else []

    def freeze_all(self)->None:
        for p in self.parameters(): p.requires_grad_(False)

    def unfreeze_last_units(self,n:int)->list[nn.Module]:
        n=max(0,int(n)); units=self.backbone_units()
        if n<=0:return []
        if not units: raise RuntimeError(f"Backbone {self.model_name!r} does not expose transformer blocks or ConvNeXt stages")
        selected=units[-min(n,len(units)):]
        for u in selected:
            for p in u.parameters():p.requires_grad_(True)
        if self.is_vit:
            norm=getattr(self.model,"norm",None)
            if norm is not None:
                for p in norm.parameters():p.requires_grad_(True)
        return selected

    def set_last_units_train(self,n:int)->None:
        self.model.eval(); units=self.backbone_units()
        if n>0 and units:
            for u in units[-min(int(n),len(units)):]:u.train()
        if self.is_vit and n>0:
            norm=getattr(self.model,"norm",None)
            if norm is not None:norm.train()

    def forward(self,x:torch.Tensor)->DenseBackboneOutput:
        self._conv_dense_cache={}; self._vit_dense_cache={}
        feat=self.model.forward_features(x) if hasattr(self.model,"forward_features") else self.model(x)
        if torch.is_tensor(feat) and feat.ndim==3:return self._vit_dense(feat,x)
        if isinstance(feat,dict):
            vals=[v for v in feat.values() if torch.is_tensor(v)]
            if not vals:raise RuntimeError("Backbone returned a dict without tensors")
            feat=vals[-1]
        if isinstance(feat,(list,tuple)):feat=feat[-1]
        if torch.is_tensor(feat) and feat.ndim==4:
            final_map=self._as_nchw(feat,self.hidden_size)
            if hasattr(self.model,"forward_head"):
                global_feat=self.model.forward_head(feat,pre_logits=True)
                if global_feat.ndim!=2:global_feat=final_map.mean((2,3))
            else:global_feat=final_map.mean((2,3))
            maps=[]
            for idx,ch in zip(self.dense_stages,self.dense_channels):
                v=self._conv_dense_cache.get(idx)
                if torch.is_tensor(v):maps.append(self._as_nchw(v,ch))
            if not maps:maps=[final_map]
            return DenseBackboneOutput(global_feat,maps[-1],maps)
        if torch.is_tensor(feat) and feat.ndim==2:
            fmap=feat[:,:,None,None]; return DenseBackboneOutput(feat,fmap,[fmap])
        raise RuntimeError(f"Unsupported backbone output: {type(feat)!r} / {getattr(feat,'shape',None)}")
