from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..data.schema import PART_CLASSES, PART_SLOT_IDS
from .dinov3 import DinoV3Backbone
from .heads import ArcMarginProduct, DinoPartHead, MultiScaleSpatialFusion, PartTokenFusion, SemanticPartAttention


class VehicleFingerprintModel(nn.Module):
    """Strong global DINOv3 ReID model with optional multiscale semantic/local evidence.

    The global branch intentionally stays close to the proven V4 recipe.  The detail branch may use
    a multiscale spatial pyramid but is never required by retrieval: score-level validation can fall
    back to the untouched global embedding.  This separation is important because semantic parts
    are useful only for some view-compatible pairs.
    """
    def __init__(
        self,
        backbone: dict,
        *,
        num_classes: int = 0,
        embed_dim: int = 512,
        part_dim: int = 128,
        local_dim: int = 128,
        local_grid: tuple[int, int] = (4, 6),
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        arc_scale: float = 30.0,
        arc_margin: float = 0.35,
        part_head_dim: int = 256,
        part_head_blocks: int = 2,
        part_attention_heads: int = 8,
        part_prior_strength: float = 2.5,
        part_visibility_threshold: float = 0.35,
        part_visibility_topk: int = 3,
        detach_semantic_prior: bool = True,
        part_dropout: float = 0.08,
        fusion_max_residual: float = 0.25,
        fusion_initial_gate: float = 0.01,
        enable_parts: bool = True,
        global_head_mode: str = "baseline_v4",
        multiscale_spatial: bool = False,
        spatial_fusion_dim: int = 256,
    ):
        super().__init__()
        self.backbone = DinoV3Backbone(**backbone)
        dg = self.backbone.hidden_size
        self.enable_parts = bool(enable_parts)
        self.global_head_mode = str(global_head_mode)
        self.part_dropout = float(part_dropout)
        self.multiscale_spatial = bool(multiscale_spatial and len(self.backbone.dense_channels) > 1)
        if self.multiscale_spatial:
            self.spatial_fusion = MultiScaleSpatialFusion(self.backbone.dense_channels, int(spatial_fusion_dim))
            dd = int(spatial_fusion_dim)
        else:
            self.spatial_fusion = None
            dd = self.backbone.dense_size

        self.global_proj = nn.Linear(dg, embed_dim, bias=False)
        nn.init.normal_(self.global_proj.weight, std=0.01)
        self.global_bn = nn.BatchNorm1d(embed_dim)
        self.global_bn.bias.requires_grad_(False)

        self.part_head = DinoPartHead(dd, len(PART_CLASSES), hidden_dim=part_head_dim, blocks=part_head_blocks)
        self.part_attention = SemanticPartAttention(
            dd, embed_dim, len(PART_SLOT_IDS), heads=part_attention_heads,
            prior_strength=part_prior_strength,
            visibility_threshold=part_visibility_threshold,
            visibility_topk=part_visibility_topk,
            detach_semantic_prior=detach_semantic_prior,
        )
        self.fusion = PartTokenFusion(
            embed_dim, len(PART_SLOT_IDS), layers=fusion_layers, heads=fusion_heads,
            max_residual=fusion_max_residual, initial_gate=fusion_initial_gate,
        )
        self.part_out = nn.Linear(embed_dim, part_dim, bias=False)
        self.local_proj = nn.Conv2d(dd, local_dim, kernel_size=1, bias=False)
        self.local_grid = tuple(local_grid)
        self.classifier = ArcMarginProduct(embed_dim, num_classes, arc_scale, arc_margin) if num_classes > 0 else None

    def _global_token(self, global_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.global_proj(global_feat); bn = self.global_bn(raw)
        return bn, F.normalize(bn, dim=-1)

    def spatial_features(self, backbone_output) -> torch.Tensor:
        if self.spatial_fusion is not None:
            maps = backbone_output.feature_maps or [backbone_output.feature_map]
            return self.spatial_fusion(maps)
        return backbone_output.feature_map

    def forward(self, image: torch.Tensor, labels: torch.Tensor | None = None, visibility_override: torch.Tensor | None = None):
        b = self.backbone(image)
        global_token, z_global = self._global_token(b.global_feat)
        spatial = self.spatial_features(b)

        local_map = self.local_proj(spatial)
        local = F.adaptive_avg_pool2d(local_map, self.local_grid).flatten(2).transpose(1, 2)
        local = F.normalize(local, dim=-1); z_local = F.normalize(local.mean(dim=1), dim=-1)

        if not self.enable_parts:
            B, _, H, W = spatial.shape
            part_logits = torch.zeros((B, len(PART_CLASSES), H, W), device=image.device, dtype=spatial.dtype)
            part_probs = torch.sigmoid(part_logits)
            vis = torch.zeros((B, len(PART_SLOT_IDS)), device=image.device, dtype=torch.bool)
            z_parts = torch.zeros((B, len(PART_SLOT_IDS), self.part_out.out_features), device=image.device, dtype=global_token.dtype)
            logits = self.classifier(z_global, labels) if self.classifier is not None else None
            return {
                "z_global": z_global, "z_fused": z_global, "z_local": z_local,
                "parts": z_parts, "visibility": vis, "visibility_score": vis.float(),
                "local": local, "logits": logits, "part_logits": part_logits, "part_probs": part_probs,
            }

        part_logits = self.part_head(spatial); part_probs = torch.sigmoid(part_logits)
        raw_parts, vis, vis_score = self.part_attention(spatial, part_probs[:, PART_SLOT_IDS])
        teacher_vis = visibility_override.bool() if visibility_override is not None else torch.zeros_like(vis)
        vis_fusion = vis | teacher_vis
        if self.training and self.part_dropout > 0:
            keep = torch.rand_like(vis_score) >= self.part_dropout
            vis_fusion = teacher_vis | (vis & keep)

        fused, _ = self.fusion(global_token, raw_parts, vis_fusion)
        z_fused = F.normalize(fused, dim=-1); z_parts = F.normalize(self.part_out(raw_parts), dim=-1)
        logits = self.classifier(z_fused, labels) if self.classifier is not None else None
        return {
            "z_global": z_global, "z_fused": z_fused, "z_local": z_local,
            "parts": z_parts, "visibility": vis, "visibility_score": vis_score,
            "local": local, "logits": logits, "part_logits": part_logits, "part_probs": part_probs,
        }
