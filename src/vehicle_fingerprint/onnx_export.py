from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .data.schema import PART_SLOT_IDS
from .pairs import load_reranker

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


class FullFeatureWrapper(nn.Module):
    """Stable ONNX tuple interface for the complete v0.9.4 feature extractor."""

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
            B, _, _, _ = spatial.shape
            vis = torch.zeros((B, len(PART_SLOT_IDS)), device=image.device, dtype=torch.bool)
            z_parts = torch.zeros(
                (B, len(PART_SLOT_IDS), m.part_out.out_features),
                device=image.device,
                dtype=global_token.dtype,
            )
            return (
                z_global,
                z_global,
                z_local,
                z_parts,
                vis.to(z_global.dtype),
                vis.to(z_global.dtype),
                local,
            )

        part_logits = m.part_head(spatial)
        part_probs = torch.sigmoid(part_logits)
        raw_parts, vis, vis_score = m.part_attention(spatial, part_probs[:, PART_SLOT_IDS])
        fused, _ = m.fusion(global_token, raw_parts, vis)
        z_fused = F.normalize(fused, dim=-1)
        z_parts = F.normalize(m.part_out(raw_parts), dim=-1)
        return (
            z_fused,
            z_global,
            z_local,
            z_parts,
            vis.to(z_global.dtype),
            vis_score,
            local,
        )


class PairRerankerExportWrapper(nn.Module):
    def __init__(self, reranker: nn.Module):
        super().__init__()
        self.reranker = reranker

    def forward(self, pair_features: torch.Tensor):
        return torch.sigmoid(self.reranker(pair_features))


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
    kwargs = dict(
        input_names=["input"],
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        opset_version=int(opset),
        do_constant_folding=True,
    )
    try:
        with torch.inference_mode():
            # torch 2.5+ accepts dynamo=False; older supported target runtimes do not.
            try:
                torch.onnx.export(wrapper, (x,), str(out), dynamo=False, **kwargs)
            except TypeError as e:
                if "dynamo" not in str(e):
                    raise
                torch.onnx.export(wrapper, (x,), str(out), **kwargs)
    finally:
        if old_mha_fastpath is not None:
            mha_backend.set_fastpath_enabled(old_mha_fastpath)
    try:
        import onnx
        model = onnx.load(str(out), load_external_data=True)
        onnx.checker.check_model(model)
    except ImportError:
        pass
    return out


def torch_dtype_for_precision(precision: str) -> torch.dtype:
    p = str(precision).lower()
    if p == "fp32":
        return torch.float32
    if p == "fp16":
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    raise ValueError("precision must be fp32/fp16/bf16")


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
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=["pair_features"],
        output_names=["match_probability"],
        dynamic_axes={"pair_features": {0: "pairs"}, "match_probability": {0: "pairs"}},
        opset_version=int(opset),
        do_constant_folding=True,
    )
    with torch.inference_mode():
        try:
            torch.onnx.export(m, (x,), str(out), dynamo=False, **kwargs)
        except TypeError as e:
            if "dynamo" not in str(e):
                raise
            torch.onnx.export(m, (x,), str(out), **kwargs)
    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(out), load_external_data=True))
    except ImportError:
        pass
    return out, input_dim
