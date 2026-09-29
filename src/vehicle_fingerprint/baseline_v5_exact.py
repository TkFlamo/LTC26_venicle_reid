from __future__ import annotations

"""Thin adapter around the user-supplied ``vehicle_reid_v5_official`` trainer.

The optimization/model/sampler/loss/scheduler/augmentation/checkpoint-selection code is
executed by the vendored V5 source itself.  This module only supplies a deterministic
pre-split train/validation manifest, optionally changes the DINOv3 architecture while
keeping the chosen V5 launcher profile, and allows a different input resolution.

v0.9.4 uses the exact V5 ``base`` profile for both ConvNeXt-Base and ViT-Base.  The
ConvNeXt branch therefore matches the supplied base launcher except for the externally
fixed split and 384x576 input.  The ViT branch keeps the same V5 base optimization recipe
and only swaps the DINOv3 backbone; a timm dynamic-size constructor shim is required for
non-default ViT resolution.

For final refit, V5's optimization loop is still used for a fixed number of epochs.  A
small diagnostic validation slice is passed only because the original V5 loop always
performs validation; it never selects epochs/checkpoints in that mode and ``last.pt`` is
used downstream.
"""

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "third_party" / "vehicle_reid_v5_official"


def _load_vendor() -> tuple[ModuleType, ModuleType]:
    p = str(VENDOR)
    if p not in sys.path:
        sys.path.insert(0, p)
    v5 = importlib.import_module("vehicle_reid_v5")
    launch = importlib.import_module("train_best_v5")
    return v5, launch


def _replace_opt(argv: list[str], key: str, value: str) -> list[str]:
    out = list(argv)
    if key in out:
        i = len(out) - 1 - out[::-1].index(key)
        if i + 1 < len(out):
            out[i + 1] = str(value)
            return out
    out += [key, str(value)]
    return out


def _drop_flag(argv: list[str], flag: str) -> list[str]:
    return [x for x in argv if x != flag]


def exact_v5_profile_argv(
    *,
    profile: str,
    model_name: str,
    height: int = 384,
    width: int = 576,
    out: str,
    images_dir: str,
    train_csv: str,
    device: str = "0",
    precision: str = "bf16",
    epochs: int | None = None,
    patience: int | None = None,
    train_microbatch: int | None = None,
    official_val: bool = True,
    init_checkpoint: str | Path | None = None,
    plate_mask_prob: float | None = None,
) -> list[str]:
    """Build arguments from an exact supplied V5 launcher profile.

    ``train_microbatch`` is activation-memory-only in V5: the complete P×K metric batch
    is concatenated before BN/ArcFace/Triplet/Circle, so changing it does not alter the
    effective metric-learning batch.
    """
    _v5, launch = _load_vendor()
    if profile not in launch.PROFILES:
        raise ValueError(f"Unknown V5 profile {profile!r}; available={sorted(launch.PROFILES)}")
    argv = list(launch.COMMON) + list(launch.PROFILES[profile])
    argv = _replace_opt(argv, "--model", model_name)
    argv = _replace_opt(argv, "--height", str(int(height)))
    argv = _replace_opt(argv, "--width", str(int(width)))
    argv = _replace_opt(argv, "--out", out)
    argv = _replace_opt(argv, "--data", str(ROOT))
    dev = str(device)
    if dev.isdigit():
        dev = f"cuda:{dev}"
    argv += ["--images-dir", images_dir, "--train-csv", train_csv, "--device", dev, "--precision", str(precision)]
    if epochs is not None:
        argv = _replace_opt(argv, "--epochs", str(int(epochs)))
    if patience is not None:
        argv = _replace_opt(argv, "--patience", str(int(patience)))
    if train_microbatch is not None:
        argv = _replace_opt(argv, "--train-microbatch", str(int(train_microbatch)))
    if init_checkpoint:
        argv = _replace_opt(argv, "--init-checkpoint", str(init_checkpoint))
    if plate_mask_prob is not None:
        argv = _replace_opt(argv, "--plate-mask-prob", str(float(plate_mask_prob)))
    if not official_val:
        argv = _drop_flag(argv, "--official-val")
        argv += ["--no-official-val"]
    return argv


def exact_v5_base_argv(**kwargs) -> list[str]:
    return exact_v5_profile_argv(profile="base", **kwargs)


def exact_v5_large_argv(**kwargs) -> list[str]:
    """Backward-compatible helper retained for old v0.9.3 tests/tools."""
    return exact_v5_profile_argv(profile="large", **kwargs)


def _read_v5_manifest(v5: ModuleType, path: str | Path) -> pd.DataFrame:
    return v5.read_manifest(Path(path), True)


def _validate_disjoint(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    tr_ids = set(train_df.vehicle_id.astype(str))
    va_ids = set(val_df.vehicle_id.astype(str))
    overlap = tr_ids & va_ids
    if overlap:
        raise ValueError(f"V5 exact presplit requires identity-disjoint train/val; overlap={len(overlap)}")


def _diagnostic_val(train_df: pd.DataFrame, n_ids: int = 12) -> pd.DataFrame:
    """Small deterministic V5-compatible validation used only during fixed-epoch final refit."""
    chosen = []
    for pid, g in train_df.groupby(train_df.vehicle_id.astype(str), sort=True):
        if len(g) >= 2:
            chosen.append(str(pid))
        if len(chosen) >= int(n_ids):
            break
    if not chosen:
        raise RuntimeError("Cannot build final-refit diagnostic validation: no identity has >=2 rows")
    return train_df[train_df.vehicle_id.astype(str).isin(chosen)].reset_index(drop=True)


def train_v5_exact_presplit(
    *,
    train_manifest: str | Path,
    val_manifest: str | Path | None,
    images_dir: str | Path,
    out_dir: str | Path,
    model_name: str,
    profile: str = "base",
    height: int = 384,
    width: int = 576,
    device: str = "0",
    precision: str = "bf16",
    epochs: int | None = None,
    final_refit: bool = False,
    train_microbatch: int | None = None,
    init_checkpoint: str | Path | None = None,
    plate_mask_prob: float | None = None,
) -> Path:
    v5, _launch = _load_vendor()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr = _read_v5_manifest(v5, train_manifest)
    if final_refit:
        va = _diagnostic_val(tr)
    else:
        if val_manifest is None:
            raise ValueError("val_manifest is required unless final_refit=True")
        va = _read_v5_manifest(v5, val_manifest)
        _validate_disjoint(tr, va)

    argv = exact_v5_profile_argv(
        profile=profile,
        model_name=model_name,
        height=height,
        width=width,
        out=str(out),
        images_dir=str(images_dir),
        train_csv=str(train_manifest),
        device=device,
        precision=precision,
        epochs=epochs,
        patience=0 if final_refit else None,
        train_microbatch=train_microbatch,
        official_val=not final_refit,
        init_checkpoint=init_checkpoint,
        plate_mask_prob=plate_mask_prob,
    )
    args = v5.parser().parse_args(argv)
    if final_refit:
        args.val_folds = 1
        args.official_val = False
        args.patience = 0

    val_ids = sorted(va.vehicle_id.astype(str).unique().tolist())
    original_split = v5.id_split_v5
    original_create_model = v5.timm.create_model if getattr(v5, "timm", None) is not None else None

    def fixed_split(_all_df, _frac, _seed):
        return tr.copy(), va.copy(), list(val_ids)

    def create_model_resolution_aware(name, *ca, **kw):
        if str(name) == str(args.model) and str(name).lower().startswith("vit"):
            kw.setdefault("img_size", (int(args.height), int(args.width)))
            kw.setdefault("dynamic_img_size", True)
            kw.setdefault("dynamic_img_pad", True)
        return original_create_model(name, *ca, **kw)

    v5.id_split_v5 = fixed_split
    if original_create_model is not None:
        v5.timm.create_model = create_model_resolution_aware
    try:
        print("=" * 92)
        print(f"V5-EXACT GLOBAL TRAINER (vendored vehicle_reid_v5_official profile={profile})")
        print(f"model={args.model} resolution={args.height}x{args.width} P{args.p}xK{args.k}={args.p*args.k}")
        print(f"epochs={args.epochs} stage2={args.stage2_epoch} milestones={args.lr_milestones} sampler={args.sampler}")
        print(f"lr_backbone={args.lr_backbone:g} lr_head={args.lr_head:g} microbatch={args.train_microbatch}")
        print(f"train_rows={len(tr)} val_rows={len(va)} final_refit={final_refit}")
        if init_checkpoint:
            print(f"init_checkpoint={init_checkpoint}")
        if plate_mask_prob is not None:
            print(f"plate_mask_prob={float(plate_mask_prob):.3f}")
        print("V5 code/recipe are unchanged; only fixed split, backbone swap (ViT branch), input resolution and memory-only microbatch may differ.")
        print("=" * 92)
        v5.run_train(args)
    finally:
        v5.id_split_v5 = original_split
        if original_create_model is not None:
            v5.timm.create_model = original_create_model

    audit = {
        "vendor_vehicle_reid_v5_sha256": "1d8a2872f5f6e46cbabee3f67183ebd1f25437252335752033c7a6b21b83fc4c",
        "vendor_train_best_v5_sha256": "4a682f415962ef6c7483712e8b9f42463da577a03f1836154e6ba40a96f5e688",
        "v5_profile": profile,
        "model": args.model,
        "height": args.height,
        "width": args.width,
        "p": args.p,
        "k": args.k,
        "effective_batch": args.p * args.k,
        "train_microbatch": args.train_microbatch,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "stage2_epoch": args.stage2_epoch,
        "lr_milestones": args.lr_milestones,
        "lr_gamma": args.lr_gamma,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "sampler": args.sampler,
        "camera_aware_triplet": args.camera_aware_triplet,
        "cross_camera_triplet": args.cross_camera_triplet,
        "same_camera_hard_neg": args.same_camera_hard_neg,
        "same_camera_neg_weight": args.same_camera_neg_weight,
        "ce_weight": args.ce_weight,
        "triplet_weight": args.triplet_weight,
        "circle_weight": args.circle_weight,
        "stage2_ce_weight": args.stage2_ce_weight,
        "stage2_triplet_weight": args.stage2_triplet_weight,
        "stage2_circle_weight": args.stage2_circle_weight,
        "val_folds": args.val_folds,
        "val_open_set_ratio": args.val_open_set_ratio,
        "official_val": args.official_val,
        "final_refit": bool(final_refit),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "plate_mask_prob": float(args.plate_mask_prob),
        "vit_dynamic_size_constructor_override": bool(str(args.model).lower().startswith("vit")),
    }
    (out / "v5_exact_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return out / ("last.pt" if final_refit else "best.pt")
