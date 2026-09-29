from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .features import load_inference_model
from .data.schema import PART_SLOT_IDS
from .pairs import load_reranker
from .v5_checkpoint import convert_v5_checkpoint


GLOBAL_OUTPUTS = ("embedding",)
FULL_OUTPUTS = (
    "z_fused",
    "z_global",
    "z_local",
    "parts",
    "visibility",
    "visibility_score",
    "local",
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def _native_to_project_if_needed(native: Path, project: Path) -> Path | None:
    if project.is_file():
        return project
    if native.is_file():
        project.parent.mkdir(parents=True, exist_ok=True)
        return convert_v5_checkpoint(native, project)
    return None


def discover_v094_global_checkpoints(project_root: str | Path, run_root: str | Path | None = None) -> dict[str, Path]:
    """Find the two global checkpoints used by the v0.9.4 Base experiment.

    Preference is deliberately direct-DINO target training, because the controlled VeRi
    experiment can be worse and the ensemble result quoted for this branch is the global
    ViT + ConvNeXt first-stage ensemble.  Explicit --vit-checkpoint/--convnext-checkpoint
    still overrides discovery in the CLI.
    """
    root = Path(project_root).resolve()
    roots: list[Path] = []
    if run_root:
        roots.append((root / run_root).resolve() if not Path(run_root).is_absolute() else Path(run_root))
    roots += [
        root / "runs/full_single_v094_v5exact_base",
        root / "runs/full_single_v094_v5exact_base_veri",
    ]

    out: dict[str, Path] = {}
    for key, bname in (("vit", "vit_base"), ("convnext", "convnext_base")):
        candidates: list[Path] = []
        native_candidates: list[tuple[Path, Path]] = []
        for rr in roots:
            candidates += [
                rr / "global_v5_exact_direct" / bname / "384x576/project_best.pt",
                rr / "global_v5_exact" / bname / "384x576/project_best.pt",
            ]
            native_candidates += [
                (
                    rr / "global_v5_exact_direct" / bname / "384x576/best.pt",
                    rr / "global_v5_exact_direct" / bname / "384x576/project_best.pt",
                ),
                (
                    rr / "global_v5_exact" / bname / "384x576/best.pt",
                    rr / "global_v5_exact" / bname / "384x576/project_best.pt",
                ),
            ]
        if key == "vit":
            candidates += [root / "artifacts/baseline/vit_base_model.pt"]
        else:
            candidates += [root / "artifacts/baseline/model.pt"]

        p = _first_existing(candidates)
        if p is None:
            for native, project in native_candidates:
                p = _native_to_project_if_needed(native, project)
                if p is not None:
                    break
        if p is not None:
            out[key] = p
    return out


def discover_v094_full_deployment(project_root: str | Path, config: dict | None = None) -> dict[str, Path | str | None]:
    root = Path(project_root).resolve()
    candidates: list[Path] = []
    if config:
        deploy = config.get("paths", {}).get("deploy")
        if deploy:
            p = Path(deploy)
            candidates.append(p if p.is_absolute() else root / p)
    candidates += [
        root / "deploy/models_v094_v5exact_base",
        root / "deploy/models_v094_v5exact_base_veri",
    ]
    for d in candidates:
        reid = d / "reid.pt"
        if reid.is_file():
            return {
                "dir": d,
                "checkpoint": reid,
                "reranker": d / "reranker.pt" if (d / "reranker.pt").is_file() else None,
                "refusal": d / "refusal.json" if (d / "refusal.json").is_file() else None,
                "recipe": d / "retrieval_recipe.json" if (d / "retrieval_recipe.json").is_file() else None,
            }
    return {"dir": None, "checkpoint": None, "reranker": None, "refusal": None, "recipe": None}


class GlobalEmbeddingWrapper(nn.Module):
    """Export only the exact global branch used by the first-stage ensemble.

    This deliberately avoids part/local heads.  It calls the timm backbone directly and reproduces
    the v0.9.4 global pooling + projection + BN + L2 normalization.  Fixed H/W keeps ViT patch-token
    selection static while batch remains dynamic for TensorRT optimization profiles.
    """

    def __init__(self, model: nn.Module, image_size: Sequence[int] = (384, 576)):
        super().__init__()
        self.backbone_wrapper = model.backbone
        self.backbone = model.backbone.model
        self.global_proj = model.global_proj
        self.global_bn = model.global_bn
        self.family = str(model.backbone.family)
        self.global_feature_mode = str(model.backbone.global_feature_mode)
        self.image_h, self.image_w = map(int, image_size)
        ph, pw = model.backbone.patch_size
        self.patch_h, self.patch_w = int(ph), int(pw)
        self.num_patches = math.ceil(self.image_h / self.patch_h) * math.ceil(self.image_w / self.patch_w)

        # Dense hooks are useful for the research model but are unnecessary for the global-only
        # engine. Removing them reduces Python-side export state and avoids retaining intermediate
        # tensors during eager equivalence checks. It does not change backbone computation.
        for name in ("_conv_dense_handles", "_vit_dense_handles"):
            handles = getattr(self.backbone_wrapper, name, [])
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
            setattr(self.backbone_wrapper, name, [])

    def _global_feat(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x) if hasattr(self.backbone, "forward_features") else self.backbone(x)
        if isinstance(feat, dict):
            vals = [v for v in feat.values() if torch.is_tensor(v)]
            feat = vals[-1]
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if self.family == "vit" or (torch.is_tensor(feat) and feat.ndim == 3):
            if feat.ndim != 3 or feat.shape[1] < self.num_patches:
                raise RuntimeError(f"Unexpected ViT tokens {tuple(feat.shape)}")
            if self.global_feature_mode == "v5_avg":
                return feat[:, -self.num_patches :, :].mean(1)
            if hasattr(self.backbone, "forward_head"):
                y = self.backbone.forward_head(feat, pre_logits=True)
                if y.ndim == 2:
                    return y
            return feat[:, -self.num_patches :, :].mean(1)
        if torch.is_tensor(feat) and feat.ndim == 4:
            if hasattr(self.backbone, "forward_head"):
                y = self.backbone.forward_head(feat, pre_logits=True)
                if y.ndim == 2:
                    return y
            # Support both NCHW and NHWC outputs.
            if feat.shape[1] == getattr(self.backbone_wrapper, "hidden_size", feat.shape[1]):
                return feat.mean((2, 3))
            return feat.mean((1, 2))
        if torch.is_tensor(feat) and feat.ndim == 2:
            return feat
        raise RuntimeError(f"Unsupported backbone output {type(feat)!r}/{getattr(feat, 'shape', None)}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self._global_feat(image)
        z = self.global_bn(self.global_proj(feat))
        return F.normalize(z, dim=-1)


class FullFeatureWrapper(nn.Module):
    """Stable tuple interface for exporting the complete neural feature extractor.

    The production model uses ``adaptive_avg_pool2d(local_map, local_grid)``.
    The legacy TorchScript ONNX exporter cannot lower that operator when the
    upstream ViT/spatial graph carries symbolic H/W, even though v0.9.4 fixes
    the image size to 384x576. For the fixed ViT patch map, adaptive pooling is
    exactly representable by ordinary AvgPool2d and is exported in that form.
    """

    def __init__(self, model: nn.Module, image_size: Sequence[int] = (384, 576)):
        super().__init__()
        self.model = model
        self.image_size = tuple(map(int, image_size))
        self.local_grid = tuple(map(int, getattr(model, "local_grid", (4, 6))))
        self._fixed_pool_kernel: tuple[int, int] | None = None
        bb = getattr(model, "backbone", None)
        patch = getattr(bb, "patch_size", None)
        if bool(getattr(bb, "is_vit", False)) and patch is not None:
            ph, pw = map(int, patch)
            h, w = self.image_size
            src_h = max(1, (h + ph - 1) // ph)
            src_w = max(1, (w + pw - 1) // pw)
            gh, gw = self.local_grid
            if src_h % gh == 0 and src_w % gw == 0:
                self._fixed_pool_kernel = (src_h // gh, src_w // gw)

    def _local_pool(self, local_map: torch.Tensor) -> torch.Tensor:
        if self._fixed_pool_kernel is not None:
            return F.avg_pool2d(
                local_map,
                kernel_size=self._fixed_pool_kernel,
                stride=self._fixed_pool_kernel,
            )
        return F.adaptive_avg_pool2d(local_map, self.local_grid)

    def forward(self, image: torch.Tensor):
        m = self.model
        b = m.backbone(image)
        global_token, z_global = m._global_token(b.global_feat)
        spatial = m.spatial_features(b)

        local_map = m.local_proj(spatial)
        local = self._local_pool(local_map).flatten(2).transpose(1, 2)
        local = F.normalize(local, dim=-1)
        z_local = F.normalize(local.mean(dim=1), dim=-1)

        if not m.enable_parts:
            B, _, H, W = spatial.shape
            vis = torch.zeros((B, len(PART_SLOT_IDS)), device=image.device, dtype=torch.bool)
            z_parts = torch.zeros(
                (B, len(PART_SLOT_IDS), m.part_out.out_features),
                device=image.device, dtype=global_token.dtype,
            )
            return (
                z_global, z_global, z_local, z_parts,
                vis.to(z_global.dtype), vis.to(z_global.dtype), local,
            )

        part_logits = m.part_head(spatial)
        part_probs = torch.sigmoid(part_logits)
        raw_parts, vis, vis_score = m.part_attention(spatial, part_probs[:, PART_SLOT_IDS])
        fused, _ = m.fusion(global_token, raw_parts, vis)
        z_fused = F.normalize(fused, dim=-1)
        z_parts = F.normalize(m.part_out(raw_parts), dim=-1)
        return (
            z_fused, z_global, z_local, z_parts,
            vis.to(z_global.dtype), vis_score, local,
        )


class PairRerankerExportWrapper(nn.Module):
    def __init__(self, reranker: nn.Module):
        super().__init__()
        self.reranker = reranker

    def forward(self, pair_features: torch.Tensor):
        return torch.sigmoid(self.reranker(pair_features))


def load_export_wrapper(checkpoint: str | Path, mode: str, device: str = "cpu"):
    model, dev, mcfg = load_inference_model(checkpoint, device=device)
    model.eval()
    size = mcfg.get("preprocess", {}).get("image_size", [384, 576])
    if mode == "global":
        return GlobalEmbeddingWrapper(model, size).to(dev).eval(), dev, mcfg, tuple(GLOBAL_OUTPUTS)
    if mode == "full":
        return FullFeatureWrapper(model).to(dev).eval(), dev, mcfg, tuple(FULL_OUTPUTS)
    raise ValueError(f"Unknown export mode: {mode}")


def pytorch_wrapper_equivalence(checkpoint: str | Path, wrapper: nn.Module, device: torch.device, image_size=(384, 576)) -> dict:
    """For global export, compare wrapper against the ordinary project forward once."""
    model, _, _ = load_inference_model(checkpoint, device=str(device))
    model.eval()
    x = torch.randn(2, 3, int(image_size[0]), int(image_size[1]), device=device)
    with torch.inference_mode():
        ref = model(x)["z_global"]
        out = wrapper(x)
    cos = F.cosine_similarity(ref.float(), out.float(), dim=-1)
    return {
        "max_abs": float((ref.float() - out.float()).abs().max().item()),
        "mean_cosine": float(cos.mean().item()),
        "min_cosine": float(cos.min().item()),
    }


def export_onnx(
    wrapper: nn.Module,
    out_path: str | Path,
    *,
    image_size: Sequence[int] = (384, 576),
    output_names: Sequence[str],
    opset: int = 18,
    dynamic_batch: bool = True,
    input_dtype: torch.dtype = torch.float32,
    export_device: str | torch.device = "cpu",
    model_dtype: torch.dtype | None = None,
) -> Path:
    """Export a feature wrapper to ONNX.

    For TensorRT 11+ reduced-precision deployment we deliberately export the
    PyTorch graph *already typed* as FP16/BF16 on CUDA. TensorRT 11 networks
    are strongly typed, so this avoids a second whole-graph ModelOpt AutoCast
    pass (which is fragile on timm/DINOv3 dynamic-shape subgraphs).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device(export_device)
    wrapper = wrapper.to(dev).eval()
    if model_dtype is not None:
        wrapper = wrapper.to(dtype=model_dtype)
    h, w = map(int, image_size)
    x = torch.randn(1, 3, h, w, dtype=input_dtype, device=dev)
    dynamic_axes = {"input": {0: "batch"}} if dynamic_batch else None
    if dynamic_batch:
        dynamic_axes.update({name: {0: "batch"} for name in output_names})
    mha_backend = getattr(torch.backends, "mha", None)
    old_mha_fastpath = None
    if mha_backend is not None and hasattr(mha_backend, "get_fastpath_enabled"):
        old_mha_fastpath = bool(mha_backend.get_fastpath_enabled())
        mha_backend.set_fastpath_enabled(False)
    try:
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (x,),
                str(out),
                input_names=["input"],
                output_names=list(output_names),
                dynamic_axes=dynamic_axes,
                opset_version=int(opset),
                do_constant_folding=True,
                dynamo=False,
            )
    finally:
        if old_mha_fastpath is not None:
            mha_backend.set_fastpath_enabled(old_mha_fastpath)
    try:
        import onnx
        m = onnx.load(str(out))
        onnx.checker.check_model(m)
    except ImportError:
        pass
    return out


def _torch_dtype_for_precision(precision: str) -> torch.dtype:
    p = str(precision).lower()
    if p == "fp32":
        return torch.float32
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    raise ValueError("precision must be fp32/fp16/bf16")


def _onnx_input_elem_type(onnx_path: str | Path, input_name: str) -> int | None:
    try:
        import onnx
    except Exception:
        return None
    m = onnx.load(str(onnx_path), load_external_data=False)
    for value in m.graph.input:
        if value.name == input_name:
            return int(value.type.tensor_type.elem_type)
    return None


def _onnx_input_matches_precision(onnx_path: str | Path, input_name: str, precision: str) -> bool:
    try:
        import onnx
    except Exception:
        return False
    want = {
        "fp32": int(onnx.TensorProto.FLOAT),
        "fp16": int(onnx.TensorProto.FLOAT16),
        "bf16": int(onnx.TensorProto.BFLOAT16),
    }.get(str(precision).lower())
    if want is None:
        return False
    return _onnx_input_elem_type(onnx_path, input_name) == want

def export_reranker_onnx(
    reranker_checkpoint: str | Path,
    out_path: str | Path,
    opset: int = 18,
    *,
    input_dtype: torch.dtype = torch.float32,
    export_device: str | torch.device = "cpu",
    model_dtype: torch.dtype | None = None,
) -> tuple[Path, int]:
    ck = torch.load(reranker_checkpoint, map_location="cpu", weights_only=False)
    input_dim = int(ck["input_dim"])
    dev = torch.device(export_device)
    m = PairRerankerExportWrapper(load_reranker(reranker_checkpoint, str(dev))).to(dev).eval()
    if model_dtype is not None:
        m = m.to(dtype=model_dtype)
    x = torch.randn(1, input_dim, dtype=input_dtype, device=dev)
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            m, (x,), str(out), input_names=["pair_features"], output_names=["match_probability"],
            dynamic_axes={"pair_features": {0: "pairs"}, "match_probability": {0: "pairs"}},
            opset_version=int(opset), do_constant_folding=True, dynamo=False,
        )
    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(out)))
    except ImportError:
        pass
    return out, input_dim

def find_trtexec(explicit: str | None = None) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.getenv("TRTEXEC")
    if env:
        candidates.append(env)
    candidates += [
        shutil.which("trtexec"),
        "/opt/tensorrt/bin/trtexec",
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
    ]
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return str(Path(c))
    raise FileNotFoundError(
        "trtexec not found. Set TRTEXEC=/path/to/trtexec or use an NVIDIA TensorRT container."
    )


def _trt_major_version(trt) -> int:
    try:
        return int(str(getattr(trt, "__version__", "0")).split(".", 1)[0])
    except Exception:
        return 0


def _builder_precision_flag(trt, precision: str):
    """Return a legacy TRT <=10 BuilderFlag or None for TRT 11+ strong typing."""
    p = str(precision).lower()
    if p == "fp32":
        return None
    name = {"fp16": "FP16", "bf16": "BF16"}.get(p)
    if name is None:
        raise ValueError("precision must be fp32/fp16/bf16")
    flags = getattr(trt, "BuilderFlag", None)
    return getattr(flags, name, None) if flags is not None else None


def _autocast_onnx_for_trt11(onnx_path: str | Path, precision: str) -> Path:
    """Bake FP16/BF16 mixed precision into ONNX for TensorRT 11+ strong typing."""
    p = str(precision).lower()
    if p == "fp32":
        return Path(onnx_path)
    if p not in {"fp16", "bf16"}:
        raise ValueError("precision must be fp32/fp16/bf16")
    try:
        import onnx
        from modelopt.onnx.autocast import convert_to_mixed_precision
    except Exception as e:
        raise RuntimeError(
            "TensorRT 11+ removed BuilderFlag.FP16/BF16. Reduced precision must be encoded "
            "in the ONNX graph. Install NVIDIA ModelOpt ONNX support with:\n"
            "  python -m pip install --user --extra-index-url https://pypi.nvidia.com 'nvidia-modelopt[onnx]'\n"
            "Then rerun the exporter. For CUDA 13, if that install pulls cupy-cuda12x, "
            "replace it with cupy-cuda13x. Alternatively export with --precision fp32."
        ) from e

    src = Path(onnx_path)
    dst = src.with_name(f"{src.stem}.{p}.trt11.onnx")
    # Reuse only when the converted graph is newer than its FP32 source.
    if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    print(f"[TensorRT] TRT 11+ strong typing: ModelOpt AutoCast {src.name} -> {dst.name} ({p})")
    converted = convert_to_mixed_precision(
        onnx_path=str(src),
        low_precision_type=p,
        keep_io_types=True,
        providers=["cpu"],
        opset=19 if p == "fp16" else 22,
    )
    onnx.save(converted, str(dst))
    return dst



def _validate_profile_shapes(
    network_shape: Sequence[int],
    min_shape: Sequence[int],
    opt_shape: Sequence[int],
    max_shape: Sequence[int],
    *,
    input_name: str = "input",
) -> None:
    """Validate a TRT optimization profile against parsed network dimensions.

    TensorRT requires every static network dimension to be identical in min/opt/max;
    only dimensions represented by -1 may vary at runtime.
    """
    real = tuple(map(int, network_shape))
    mn = tuple(map(int, min_shape)); op = tuple(map(int, opt_shape)); mx = tuple(map(int, max_shape))
    if not (len(real) == len(mn) == len(op) == len(mx)):
        raise RuntimeError(
            f"TensorRT profile rank mismatch for {input_name}: network={real} min={mn} opt={op} max={mx}"
        )
    for i, (r, a, b, c) in enumerate(zip(real, mn, op, mx)):
        if not (0 <= a <= b <= c):
            raise RuntimeError(
                f"TensorRT invalid profile dimension {i} for {input_name}: min/opt/max={a}/{b}/{c}"
            )
        if r != -1 and not (a == b == c == r):
            raise RuntimeError(
                f"TensorRT input {input_name} dimension {i} is static ({r}) in ONNX/network, "
                f"but profile asks for {a}/{b}/{c}. Re-export that dimension as dynamic or keep it fixed."
            )


def _set_profile_shape_compat(
    profile,
    input_name: str,
    min_shape: Sequence[int],
    opt_shape: Sequence[int],
    max_shape: Sequence[int],
):
    """Set TRT profile shapes across TensorRT Python API versions.

    TRT 11.3 documents IOptimizationProfile.set_shape() as returning None on
    success and raising ValueError on invalid input. Older bindings may return
    bool. Treat only an explicit False as failure.
    """
    mn = tuple(map(int, min_shape)); op = tuple(map(int, opt_shape)); mx = tuple(map(int, max_shape))
    try:
        result = profile.set_shape(input_name, mn, op, mx)
    except Exception as e:
        raise RuntimeError(
            f"TensorRT rejected optimization profile for {input_name}: "
            f"min={mn} opt={op} max={mx}: {type(e).__name__}: {e}"
        ) from e
    if result is False:
        raise RuntimeError(
            f"TensorRT rejected optimization profile for {input_name}: "
            f"min={mn} opt={op} max={mx}"
        )
    return result

def build_engine_with_python(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    input_name: str,
    min_shape: Sequence[int],
    opt_shape: Sequence[int],
    max_shape: Sequence[int],
    precision: str = "fp16",
    workspace_mib: int = 4096,
) -> tuple[Path, list[str]]:
    """Build a TensorRT engine through the Python Builder API.

    TensorRT <=10 uses legacy precision BuilderFlags. TensorRT 11+ is strongly
    typed only, so FP16/BF16 is first baked into the ONNX graph with NVIDIA
    ModelOpt AutoCast, following NVIDIA's 10.x -> 11.x migration guidance.
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise RuntimeError(
            "TensorRT Python bindings are not installed and trtexec was not found. "
            "Install a TensorRT package matching the CUDA major version, or provide TRTEXEC=/path/to/trtexec."
        ) from e

    p = str(precision).lower()
    if p not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32/fp16/bf16")

    original_onnx = Path(onnx_path)
    precision_flag = _builder_precision_flag(trt, p)
    strong_typed_reduced = p != "fp32" and precision_flag is None
    if strong_typed_reduced and _onnx_input_matches_precision(original_onnx, input_name, p):
        parse_onnx = original_onnx
        print(f"[TensorRT] TensorRT {getattr(trt, '__version__', '?')} strong typing: "
              f"{original_onnx.name} is already {p}-typed; ModelOpt AutoCast skipped")
    elif strong_typed_reduced:
        try:
            parse_onnx = _autocast_onnx_for_trt11(original_onnx, p)
        except Exception as e:
            raise RuntimeError(
                f"TensorRT 11+ needs a reduced-precision ONNX graph for {p}. "
                "ModelOpt AutoCast failed on this graph. Re-export the model directly in "
                f"{p} (the v0.9.4 exporter hotfix does this automatically on CUDA). "
                f"Original error: {type(e).__name__}: {e}"
            ) from e
    else:
        parse_onnx = original_onnx

    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = engine_path.with_suffix(engine_path.suffix + ".build.log")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    if builder is None:
        raise RuntimeError("TensorRT Builder could not be created; check CUDA/driver compatibility")

    explicit_flag = 0
    if hasattr(trt, "NetworkDefinitionCreationFlag") and hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        explicit_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_flag)
    parser = trt.OnnxParser(network, logger)
    ok = parser.parse(parse_onnx.read_bytes())
    if not ok:
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        log_path.write_text("\n".join(errors), encoding="utf-8", errors="replace")
        raise RuntimeError(f"TensorRT ONNX parse failed; see {log_path}")

    names = [network.get_input(i).name for i in range(network.num_inputs)]
    if input_name not in names:
        raise RuntimeError(f"ONNX/TensorRT input {input_name!r} not found; available={names}")

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mib) * 1024 * 1024)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = int(workspace_mib) * 1024 * 1024
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 5

    if precision_flag is not None:
        if p == "fp16" and not getattr(builder, "platform_has_fast_fp16", True):
            print("[TensorRT] warning: platform_has_fast_fp16=False; build may use FP32 fallbacks")
        config.set_flag(precision_flag)
    elif strong_typed_reduced:
        print(f"[TensorRT] TensorRT {getattr(trt, '__version__', '?')} strongly typed build; precision comes from {parse_onnx.name}")

    profile = builder.create_optimization_profile()
    trt_input = next(network.get_input(i) for i in range(network.num_inputs) if network.get_input(i).name == input_name)
    network_shape = tuple(int(x) for x in trt_input.shape)
    print(f"[TensorRT] input {input_name} network_shape={network_shape}; "
          f"profile min={tuple(min_shape)} opt={tuple(opt_shape)} max={tuple(max_shape)}")
    _validate_profile_shapes(network_shape, min_shape, opt_shape, max_shape, input_name=input_name)
    _set_profile_shape_compat(profile, input_name, min_shape, opt_shape, max_shape)
    profile_index = config.add_optimization_profile(profile)
    if isinstance(profile_index, int) and profile_index < 0:
        raise RuntimeError(
            f"TensorRT add_optimization_profile failed for {input_name}: index={profile_index}; "
            f"network_shape={network_shape} min={tuple(min_shape)} opt={tuple(opt_shape)} max={tuple(max_shape)}"
        )

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT Python builder returned None while building serialized engine")
    engine_path.write_bytes(bytes(serialized))
    desc = [
        "python:tensorrt.Builder",
        f"version={getattr(trt, '__version__', '?')}",
        f"onnx_source={original_onnx}",
        f"onnx_built={parse_onnx}",
        f"engine={engine_path}",
        f"input={input_name}",
        f"min={tuple(map(int, min_shape))}",
        f"opt={tuple(map(int, opt_shape))}",
        f"max={tuple(map(int, max_shape))}",
        f"precision={p}",
        f"precision_mode={'strong_typed_onnx' if strong_typed_reduced else 'legacy_builder_flag' if precision_flag is not None else 'fp32_strong_typed'}",
        f"workspace_mib={int(workspace_mib)}",
    ]
    log_path.write_text("\n".join(desc) + "\n", encoding="utf-8")
    return engine_path, desc

def build_engine_with_trtexec(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    input_name: str,
    min_shape: Sequence[int],
    opt_shape: Sequence[int],
    max_shape: Sequence[int],
    precision: str = "fp16",
    workspace_mib: int = 4096,
    trtexec: str | None = None,
    extra_args: Sequence[str] | None = None,
) -> tuple[Path, list[str]]:
    """Build with trtexec when available, otherwise use TensorRT Python Builder."""
    try:
        exe = find_trtexec(trtexec)
    except FileNotFoundError:
        print("[TensorRT] trtexec not found; falling back to TensorRT Python Builder API")
        return build_engine_with_python(
            onnx_path, engine_path, input_name=input_name,
            min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape,
            precision=precision, workspace_mib=workspace_mib,
        )

    onnx_path = Path(onnx_path); engine_path = Path(engine_path); engine_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = lambda s: "x".join(map(str, s))
    cmd = [
        exe,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={input_name}:{fmt(min_shape)}",
        f"--optShapes={input_name}:{fmt(opt_shape)}",
        f"--maxShapes={input_name}:{fmt(max_shape)}",
        f"--memPoolSize=workspace:{int(workspace_mib)}MiB",
        "--builderOptimizationLevel=5",
        "--buildOnly",
    ]
    p = str(precision).lower()
    if p == "fp16": cmd.append("--fp16")
    elif p == "bf16": cmd.append("--bf16")
    elif p != "fp32": raise ValueError("precision must be fp32/fp16/bf16")
    if extra_args: cmd.extend(map(str, extra_args))
    log_path = engine_path.with_suffix(engine_path.suffix + ".build.log")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not engine_path.is_file():
        raise RuntimeError(f"trtexec engine build failed ({proc.returncode}); see {log_path}")
    return engine_path, cmd


def build_reranker_engine_with_trtexec(
    onnx_path: str | Path, engine_path: str | Path, input_dim: int, *,
    min_pairs=1, opt_pairs=64, max_pairs=512, precision="fp16", workspace_mib=512, trtexec=None,
):
    return build_engine_with_trtexec(
        onnx_path, engine_path, input_name="pair_features",
        min_shape=(int(min_pairs), int(input_dim)),
        opt_shape=(int(opt_pairs), int(input_dim)),
        max_shape=(int(max_pairs), int(input_dim)),
        precision=precision, workspace_mib=workspace_mib, trtexec=trtexec,
    )


class TensorRTEngine:
    """Minimal TensorRT 10 runtime backed by torch CUDA tensors (no PyCUDA required)."""

    def __init__(self, engine_path: str | Path, device: str | torch.device = "cuda:0"):
        try:
            import tensorrt as trt
        except ImportError as e:
            raise RuntimeError("Python package 'tensorrt' is required for runtime benchmark") from e
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        blob = Path(engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine {engine_path}")
        if not hasattr(self.engine, "num_io_tensors"):
            raise RuntimeError("This benchmark expects TensorRT 10+ Python tensor API")
        self.context = self.engine.create_execution_context()
        self.device = torch.device(device)
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        if len(self.input_names) != 1:
            raise RuntimeError(f"Expected one input, got {self.input_names}")
        self.input_name = self.input_names[0]
        self._output_cache: dict[tuple, dict[str, torch.Tensor]] = {}

    def _torch_dtype(self, dtype):
        trt = self.trt
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        if hasattr(trt, "bfloat16"):
            mapping[trt.bfloat16] = torch.bfloat16
        if dtype not in mapping:
            raise TypeError(f"Unsupported TensorRT dtype {dtype}")
        return mapping[dtype]

    def infer(self, x: torch.Tensor, stream: torch.cuda.Stream | None = None) -> dict[str, torch.Tensor]:
        if not x.is_cuda:
            raise ValueError("TensorRT input must be a CUDA tensor")
        stream = stream or torch.cuda.current_stream(self.device)
        x = x.contiguous()
        self.context.set_input_shape(self.input_name, tuple(map(int, x.shape)))
        expected = self._torch_dtype(self.engine.get_tensor_dtype(self.input_name))
        if x.dtype != expected:
            x = x.to(dtype=expected)
        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        shapes = []
        for name in self.output_names:
            shape = tuple(map(int, self.context.get_tensor_shape(name)))
            dtype = self._torch_dtype(self.engine.get_tensor_dtype(name))
            shapes.append((name, shape, dtype))
        cache_key = (tuple(map(int, x.shape)), tuple((n, sh, str(dt)) for n, sh, dt in shapes))
        outputs = self._output_cache.get(cache_key)
        if outputs is None:
            outputs = {name: torch.empty(shape, dtype=dtype, device=self.device) for name, shape, dtype in shapes}
            self._output_cache[cache_key] = outputs
        for name, y in outputs.items():
            self.context.set_tensor_address(name, int(y.data_ptr()))
        ok = self.context.execute_async_v3(stream_handle=int(stream.cuda_stream))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        return outputs

    def benchmark(self, shape: Sequence[int], warmup=50, iterations=200) -> dict:
        dtype = self._torch_dtype(self.engine.get_tensor_dtype(self.input_name))
        x = torch.randn(tuple(map(int, shape)), device=self.device, dtype=dtype)
        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            for _ in range(int(warmup)):
                self.infer(x, stream)
        stream.synchronize()
        times = []
        for _ in range(int(iterations)):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record(stream)
                self.infer(x, stream)
                end.record(stream)
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
        a = np.asarray(times, np.float64)
        batch = int(shape[0])
        return {
            "batch": batch,
            "iterations": int(iterations),
            "mean_ms": float(a.mean()),
            "median_ms": float(np.median(a)),
            "p90_ms": float(np.percentile(a, 90)),
            "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)),
            "images_per_second": float(batch * 1000.0 / a.mean()),
        }


def weighted_ensemble_embedding(vit: torch.Tensor, convnext: torch.Tensor, convnext_weight: float) -> torch.Tensor:
    w = float(np.clip(convnext_weight, 0.0, 1.0))
    vit = F.normalize(vit.float(), dim=-1)
    convnext = F.normalize(convnext.float(), dim=-1)
    return torch.cat([math.sqrt(1.0 - w) * vit, math.sqrt(w) * convnext], dim=-1)


def topk_cosine(query: torch.Tensor, gallery: torch.Tensor, k: int = 10):
    query = F.normalize(query.float(), dim=-1)
    gallery = F.normalize(gallery.float(), dim=-1)
    scores = query @ gallery.T
    return torch.topk(scores, min(int(k), scores.shape[1]), dim=1, largest=True, sorted=True)


def summarize_times(values: Sequence[float]) -> dict:
    a = np.asarray(values, np.float64)
    if not len(a):
        return {}
    return {
        "n": int(len(a)),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def write_json(path: str | Path, obj) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
