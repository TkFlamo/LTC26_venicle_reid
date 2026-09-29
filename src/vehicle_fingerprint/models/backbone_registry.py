from __future__ import annotations

from copy import deepcopy
from typing import Any

# Canonical DINOv3 backbones exposed by the project.  The exact timm names below are the
# public DINOv3 LVD-1689M model identifiers.  Keep aliases short so experiments are easy to
# launch while checkpoints still store the exact resolved model name.
BACKBONE_PROFILES: dict[str, dict[str, Any]] = {
    "convnext_tiny": {
        "model_name": "convnext_tiny.dinov3_lvd1689m",
        "family": "convnext",
        "dense_stage": -2,
        "dense_stages": [-3, -2, -1],
        "description": "ConvNeXt-Tiny DINOv3; cheapest ConvNeXt branch",
    },
    "convnext_small": {
        "model_name": "convnext_small.dinov3_lvd1689m",
        "family": "convnext",
        "dense_stage": -2,
        "dense_stages": [-3, -2, -1],
        "description": "ConvNeXt-Small DINOv3; supplied strong-baseline default",
    },
    "convnext_base": {
        "model_name": "convnext_base.dinov3_lvd1689m",
        "family": "convnext",
        "dense_stage": -2,
        "dense_stages": [-3, -2, -1],
        "description": "ConvNeXt-Base DINOv3",
    },
    "convnext_large": {
        "model_name": "convnext_large.dinov3_lvd1689m",
        "family": "convnext",
        "dense_stage": -2,
        "dense_stages": [-3, -2, -1],
        "description": "ConvNeXt-Large DINOv3; high-capacity CNN branch",
    },
    "vit_small": {
        "model_name": "vit_small_patch16_dinov3.lvd1689m",
        "family": "vit",
        "dense_stage": None,
        # Average normalized patch tokens from multiple transformer depths for the spatial
        # branch.  Global ReID still uses timm's final pre-logits representation.
        "vit_dense_blocks": [-1, -4, -8],
        "description": "ViT-S/16 DINOv3; lightweight native patch-token branch",
    },
    "vit_base": {
        "model_name": "vit_base_patch16_dinov3.lvd1689m",
        "family": "vit",
        "dense_stage": None,
        "vit_dense_blocks": [-1, -4, -8],
        "description": "ViT-B/16 DINOv3; recommended first ViT experiment",
    },
    "vit_large": {
        "model_name": "vit_large_patch16_dinov3.lvd1689m",
        "family": "vit",
        "dense_stage": None,
        "vit_dense_blocks": [-1, -4, -8],
        "description": "ViT-L/16 DINOv3; high-capacity native patch-token branch",
    },
}

ALIASES = {
    "cn_t": "convnext_tiny", "convnext_t": "convnext_tiny", "convnexttiny": "convnext_tiny",
    "cn_s": "convnext_small", "convnext_s": "convnext_small", "convnextsmall": "convnext_small",
    "cn_b": "convnext_base", "convnext_b": "convnext_base", "convnextbase": "convnext_base",
    "cn_l": "convnext_large", "convnext_l": "convnext_large", "convnextlarge": "convnext_large",
    "vits": "vit_small", "vit_s": "vit_small", "vits16": "vit_small",
    "vitb": "vit_base", "vit_b": "vit_base", "vitb16": "vit_base",
    "vitl": "vit_large", "vit_l": "vit_large", "vitl16": "vit_large",
}


def canonical_backbone_name(name: str) -> str:
    key = str(name).strip()
    low = key.lower()
    if low in BACKBONE_PROFILES:
        return low
    if low in ALIASES:
        return ALIASES[low]
    return key  # direct timm model name; deliberately preserve case/spelling


def infer_family(model_name: str) -> str:
    low = str(model_name).lower()
    if low.startswith("vit") or "vision_transformer" in low:
        return "vit"
    if low.startswith("convnext"):
        return "convnext"
    return "generic"


def profile_config(name: str) -> dict[str, Any]:
    canonical = canonical_backbone_name(name)
    if canonical in BACKBONE_PROFILES:
        out = deepcopy(BACKBONE_PROFILES[canonical])
        out["profile"] = canonical
        return out
    # Unknown names are accepted as direct timm identifiers.  This makes the project future-proof
    # while still giving first-class aliases to the tested DINOv3 models above.
    family = infer_family(canonical)
    return {
        "profile": canonical,
        "model_name": canonical,
        "family": family,
        "dense_stage": -2 if family == "convnext" else None,
        "dense_stages": [-3, -2, -1] if family == "convnext" else None,
        "vit_dense_blocks": [-1, -4, -8] if family == "vit" else None,
        "description": "direct timm model name",
    }


def resolve_backbone_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve a model.backbone dictionary into constructor kwargs for DinoV3Backbone.

    Explicit YAML fields always win over profile defaults.  Non-constructor informational fields
    (currently ``family`` and ``description``) are removed before model construction.
    """
    cfg = deepcopy(cfg or {})
    prof = cfg.get("profile")
    base: dict[str, Any] = {}
    if prof:
        base = profile_config(str(prof))
    elif cfg.get("model_name"):
        base = profile_config(str(cfg["model_name"]))
    merged = {**base, **cfg}
    merged.pop("family", None)
    merged.pop("description", None)
    # Store canonical alias when known, but DinoV3Backbone accepts it as metadata.
    if "profile" in merged:
        merged["profile"] = canonical_backbone_name(str(merged["profile"]))
    return merged


def apply_backbone_override(cfg: dict[str, Any], backbone: str | None) -> dict[str, Any]:
    """Return a copied experiment config with a different backbone, preserving all other recipe knobs."""
    out = deepcopy(cfg)
    if not backbone:
        return out
    canonical = canonical_backbone_name(backbone)
    prof = profile_config(canonical)
    model = out.setdefault("model", {})
    old = deepcopy(model.get("backbone", {}))
    # Preserve source/checkpoint flags from the experiment but replace architecture-dependent keys.
    keep = {k: old[k] for k in ("pretrained", "checkpoint_path", "cache_dir", "img_size", "dynamic_img_size", "dynamic_img_pad", "global_feature_mode") if k in old}
    bcfg = {
        "profile": prof["profile"],
        "model_name": prof["model_name"],
        "dense_stage": prof.get("dense_stage"),
    }
    if prof.get("dense_stages") is not None:
        bcfg["dense_stages"] = list(prof["dense_stages"])
    if prof.get("vit_dense_blocks") is not None:
        bcfg["vit_dense_blocks"] = list(prof["vit_dense_blocks"])
    bcfg.update(keep)
    model["backbone"] = bcfg
    out["backbone_profile"] = prof["profile"]
    return out


def available_backbones() -> list[tuple[str, dict[str, Any]]]:
    return [(k, deepcopy(v)) for k, v in BACKBONE_PROFILES.items()]
