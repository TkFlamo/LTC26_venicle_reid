#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vehicle_fingerprint.features import load_inference_model
from vehicle_fingerprint.tensorrt_deploy import (
    FULL_OUTPUTS,
    GlobalEmbeddingWrapper,
    FullFeatureWrapper,
    build_engine_with_trtexec,
    build_reranker_engine_with_trtexec,
    discover_v094_full_deployment,
    discover_v094_global_checkpoints,
    export_onnx,
    export_reranker_onnx,
    _torch_dtype_for_precision,
    sha256_file,
    write_json,
    TensorRTEngine,
)


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def compare_global(model, wrapper, device, h, w):
    x = torch.randn(1, 3, h, w, device=device)
    model.eval(); wrapper.eval()
    with torch.inference_mode():
        ref = model(x)["z_global"].float()
        got = wrapper(x).float()
    cos = F.cosine_similarity(ref, got, dim=-1)
    return {
        "max_abs": float((ref - got).abs().max().item()),
        "mean_cosine": float(cos.mean().item()),
        "min_cosine": float(cos.min().item()),
    }




def compare_full_wrapper(model, wrapper, device, h, w):
    dtype = next(model.parameters()).dtype
    x = torch.randn(1, 3, h, w, device=device, dtype=dtype)
    model.eval(); wrapper.eval()
    with torch.inference_mode():
        ref = model(x)
        got = wrapper(x)
    refs = (
        ref["z_fused"], ref["z_global"], ref["z_local"], ref["parts"],
        ref["visibility"].to(ref["z_global"].dtype), ref["visibility_score"], ref["local"],
    )
    out = {}
    for name, a, b in zip(FULL_OUTPUTS, refs, got):
        af = a.float(); bf = b.float()
        rec = {
            "max_abs": float((af - bf).abs().max().item()),
            "mean_abs": float((af - bf).abs().mean().item()),
        }
        if af.ndim >= 2 and af.shape[-1] > 1:
            cos = F.cosine_similarity(
                af.reshape(-1, af.shape[-1]), bf.reshape(-1, bf.shape[-1]), dim=-1
            )
            rec["mean_cosine"] = float(cos.mean().item())
            rec["min_cosine"] = float(cos.min().item())
        out[name] = rec
    return out


def compare_engine_global(checkpoint, engine_path, device, image_size):
    try:
        model, _, _ = load_inference_model(checkpoint, device=str(device))
        wrapper = GlobalEmbeddingWrapper(model, image_size).to(device).eval()
        engine = TensorRTEngine(engine_path, device)
        dtype = engine._torch_dtype(engine.engine.get_tensor_dtype(engine.input_name))
        x = torch.randn(1, 3, int(image_size[0]), int(image_size[1]), device=device, dtype=dtype)
        with torch.inference_mode():
            ref = wrapper(x.float() if next(wrapper.parameters()).dtype == torch.float32 else x).float()
        got = engine.infer(x)["embedding"].float()
        torch.cuda.synchronize(device)
        cos = F.cosine_similarity(ref, got, dim=-1)
        return {
            "status": "ok",
            "max_abs": float((ref-got).abs().max().item()),
            "mean_cosine": float(cos.mean().item()),
            "min_cosine": float(cos.min().item()),
        }
    except Exception as e:
        return {"status":"skipped_or_failed", "error":repr(e)}


def compare_engine_full(checkpoint, engine_path, device, image_size):
    try:
        model, _, _ = load_inference_model(checkpoint, device=str(device))
        engine = TensorRTEngine(engine_path, device)
        dtype = engine._torch_dtype(engine.engine.get_tensor_dtype(engine.input_name))
        wrapper = FullFeatureWrapper(model, image_size).to(device=device, dtype=dtype).eval()
        x = torch.randn(1, 3, int(image_size[0]), int(image_size[1]), device=device, dtype=dtype)
        with torch.inference_mode():
            ref_tuple = wrapper(x)
        got = engine.infer(x)
        torch.cuda.synchronize(device)
        per_output = {}
        for name, ref in zip(FULL_OUTPUTS, ref_tuple):
            out = got[name]
            rf = ref.float(); of = out.float()
            rec = {
                "max_abs": float((rf - of).abs().max().item()),
                "mean_abs": float((rf - of).abs().mean().item()),
            }
            if rf.ndim >= 2 and rf.shape[-1] > 1:
                cos = F.cosine_similarity(
                    rf.reshape(-1, rf.shape[-1]), of.reshape(-1, of.shape[-1]), dim=-1
                )
                rec["mean_cosine"] = float(cos.mean().item())
                rec["min_cosine"] = float(cos.min().item())
            per_output[name] = rec
        return {"status": "ok", "outputs": per_output}
    except Exception as e:
        return {"status": "skipped_or_failed", "error": repr(e)}

def main():
    p = argparse.ArgumentParser(description="Export v0.9.4 Base global ensemble and optional full neural pipeline to ONNX/TensorRT")
    p.add_argument("--config", default="configs/full_cv_pipeline.yaml")
    p.add_argument("--run-root", default=None, help="Optional runs root; otherwise v0.9.4 candidates are auto-discovered")
    p.add_argument("--vit-checkpoint", default=None)
    p.add_argument("--convnext-checkpoint", default=None)
    p.add_argument("--full-checkpoint", default=None, help="Optional final advanced/deployment reid.pt")
    p.add_argument("--reranker", default=None, help="Optional pair reranker checkpoint")
    p.add_argument("--out", default="deploy/tensorrt_v094")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=8)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--workspace-mib", type=int, default=4096)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--trtexec", default=None)
    p.add_argument("--onnx-only", action="store_true")
    p.add_argument("--skip-full", action="store_true", help="Export only the two global first-stage engines")
    p.add_argument("--skip-reranker", action="store_true")
    p.add_argument("--device", default="0", help="Device used only for PyTorch equivalence check")
    p.add_argument("--no-equivalence-check", action="store_true")
    p.add_argument("--reuse-existing", action="store_true", help="Reuse already-built global ONNX/plan files after an optional full-export failure")
    a = p.parse_args()

    cfg_path = ROOT / a.config
    cfg = load_yaml(cfg_path) if cfg_path.is_file() else {}
    out = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    onnx_dir = out / "onnx"; engine_dir = out / "engines"
    onnx_dir.mkdir(parents=True, exist_ok=True); engine_dir.mkdir(parents=True, exist_ok=True)

    active_run_root = a.run_root or cfg.get("paths", {}).get("runs")
    found = discover_v094_global_checkpoints(ROOT, active_run_root)
    vit = Path(a.vit_checkpoint) if a.vit_checkpoint else found.get("vit")
    cn = Path(a.convnext_checkpoint) if a.convnext_checkpoint else found.get("convnext")
    if not vit or not Path(vit).is_file():
        raise SystemExit("ViT project checkpoint not found. Pass --vit-checkpoint .../project_best.pt")
    if not cn or not Path(cn).is_file():
        raise SystemExit("ConvNeXt project checkpoint not found. Pass --convnext-checkpoint .../project_best.pt")
    vit = Path(vit); cn = Path(cn)

    dep = discover_v094_full_deployment(ROOT, cfg)
    full_ckpt = Path(a.full_checkpoint) if a.full_checkpoint else dep.get("checkpoint")
    reranker = Path(a.reranker) if a.reranker else dep.get("reranker")

    dev = torch.device(f"cuda:{a.device}" if torch.cuda.is_available() else "cpu")
    try:
        import tensorrt as _trt
        trt_major = int(str(getattr(_trt, "__version__", "0")).split(".", 1)[0])
    except Exception:
        trt_major = 0
    direct_reduced_onnx = trt_major >= 11 and a.precision in {"fp16", "bf16"}
    if direct_reduced_onnx and dev.type != "cuda":
        raise SystemExit("TensorRT 11+ FP16/BF16 export requires CUDA for direct reduced-precision ONNX tracing")
    export_dtype = _torch_dtype_for_precision(a.precision) if direct_reduced_onnx else torch.float32
    export_device = dev if direct_reduced_onnx else torch.device("cpu")
    if direct_reduced_onnx:
        print(f"[ONNX] TensorRT {trt_major}: exporting model directly as {a.precision} on {export_device}; ModelOpt AutoCast will not be needed")
    manifest = {
        "schema": "vehicle-reid-v094-tensorrt-v1",
        "config": str(cfg_path.relative_to(ROOT)) if cfg_path.is_file() else None,
        "active_run_root": str(active_run_root) if active_run_root else None,
        "precision": a.precision,
        "shape_profile": {"min_batch": a.min_batch, "opt_batch": a.opt_batch, "max_batch": a.max_batch, "height": 384, "width": 576},
        "models": {},
        "notes": [
            "Global engines emit exact L2-normalized 512-D z_global embeddings used by the first-stage ViT+ConvNeXt ensemble.",
            "Spatial H/W is fixed to 384x576; only batch is dynamic.",
            "Non-neural crop/decode, similarity/top-k and refusal logic are benchmarked separately and are not TensorRT layers.",
            "TensorRT 11+ reduced precision is exported directly from PyTorch into typed FP16/BF16 ONNX; ModelOpt AutoCast is only a fallback for legacy FP32 ONNX inputs.",
        ],
    }

    for name, ckpt in (("vit", vit), ("convnext", cn)):
        model, device, mcfg = load_inference_model(ckpt, device=str(dev))
        size = tuple(map(int, mcfg.get("preprocess", {}).get("image_size", [384, 576])))
        wrapper = GlobalEmbeddingWrapper(model, size).to(device).eval()
        eq = None if a.no_equivalence_check else compare_global(model, wrapper, device, *size)
        onnx_path = onnx_dir / f"{name}_global.onnx"
        existing_engine = engine_dir / f"{name}_global.plan"
        can_reuse = bool(
            a.reuse_existing and onnx_path.is_file() and onnx_path.stat().st_size > 0
            and (a.onnx_only or (existing_engine.is_file() and existing_engine.stat().st_size > 0))
        )
        if can_reuse:
            print(f"[REUSE] {name}_global: {onnx_path}" + ("" if a.onnx_only else f" + {existing_engine}"))
        else:
            onnx_path = export_onnx(
                wrapper, onnx_path, image_size=size, output_names=("embedding",), opset=a.opset,
                input_dtype=export_dtype, export_device=export_device, model_dtype=export_dtype if direct_reduced_onnx else None,
            )
        rec = {
            "checkpoint": str(ckpt), "checkpoint_sha256": sha256_file(ckpt), "mode": "global", "image_size": list(size),
            "onnx": str(onnx_path), "onnx_sha256": sha256_file(onnx_path), "pytorch_equivalence": eq,
            "reused_existing": can_reuse,
        }
        if not a.onnx_only:
            if can_reuse:
                eng, cmd = existing_engine, ["reuse-existing"]
            else:
                eng, cmd = build_engine_with_trtexec(
                    onnx_path, existing_engine, input_name="input",
                    min_shape=(a.min_batch, 3, *size), opt_shape=(a.opt_batch, 3, *size), max_shape=(a.max_batch, 3, *size),
                    precision=a.precision, workspace_mib=a.workspace_mib, trtexec=a.trtexec,
                )
            rec.update({"engine": str(eng), "engine_sha256": sha256_file(eng), "engine_build": cmd})
            if torch.cuda.is_available():
                rec["tensorrt_equivalence"] = compare_engine_global(ckpt, eng, dev, size)
        manifest["models"][f"{name}_global"] = rec
        del wrapper, model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not a.skip_full and full_ckpt and Path(full_ckpt).is_file():
        full_ckpt = Path(full_ckpt)
        try:
            model, device, mcfg = load_inference_model(full_ckpt, device=str(export_device))
            size = tuple(map(int, mcfg.get("preprocess", {}).get("image_size", [384, 576])))
            wrapper = FullFeatureWrapper(model, size).eval()
            # Verify that the export-only fixed AvgPool path is numerically identical
            # to the ordinary production forward before serialising anything.
            full_eq = None if a.no_equivalence_check else compare_full_wrapper(model, wrapper, device, *size)
            onnx_path = export_onnx(
                wrapper, onnx_dir / "full_feature_extractor.onnx", image_size=size, output_names=FULL_OUTPUTS, opset=a.opset,
                input_dtype=export_dtype, export_device=export_device, model_dtype=export_dtype if direct_reduced_onnx else None,
            )
            rec = {
                "status": "ok",
                "checkpoint": str(full_ckpt), "checkpoint_sha256": sha256_file(full_ckpt), "mode": "full", "image_size": list(size),
                "outputs": list(FULL_OUTPUTS), "onnx": str(onnx_path), "onnx_sha256": sha256_file(onnx_path),
                "pytorch_wrapper_equivalence": full_eq,
                "retrieval_recipe": str(dep.get("recipe")) if dep.get("recipe") else None,
                "refusal": str(dep.get("refusal")) if dep.get("refusal") else None,
            }
            if not a.onnx_only:
                eng, cmd = build_engine_with_trtexec(
                    onnx_path, engine_dir / "full_feature_extractor.plan", input_name="input",
                    min_shape=(a.min_batch, 3, *size), opt_shape=(a.opt_batch, 3, *size), max_shape=(a.max_batch, 3, *size),
                    precision=a.precision, workspace_mib=a.workspace_mib, trtexec=a.trtexec,
                )
                rec.update({"engine": str(eng), "engine_sha256": sha256_file(eng), "engine_build": cmd})
                if torch.cuda.is_available() and not a.no_equivalence_check:
                    rec["tensorrt_equivalence"] = compare_engine_full(full_ckpt, eng, dev, size)
            manifest["models"]["full_feature_extractor"] = rec
        except Exception as e:
            # Full advanced extraction is optional for the first-stage ensemble benchmark.
            # Preserve already-built global engines and continue to the reranker/manifest.
            manifest["models"]["full_feature_extractor"] = {
                "status": "failed", "checkpoint": str(full_ckpt), "error": repr(e),
            }
            manifest["notes"].append(
                "Full advanced extractor export failed; global ViT/ConvNeXt engines remain valid and benchmarkable. "
                + repr(e)
            )
            print(f"[WARN] full_feature_extractor export failed but global engines are preserved: {e!r}")
    elif not a.skip_full:
        manifest["notes"].append("No deploy/reid.pt (or --full-checkpoint) was found; full advanced neural extractor was skipped.")

    if not a.skip_reranker and reranker and Path(reranker).is_file():
        reranker = Path(reranker)
        onnx_path, input_dim = export_reranker_onnx(
            reranker, onnx_dir / "pair_reranker.onnx", opset=a.opset,
            input_dtype=export_dtype, export_device=export_device, model_dtype=export_dtype if direct_reduced_onnx else None,
        )
        rec = {
            "checkpoint": str(reranker), "checkpoint_sha256": sha256_file(reranker), "input_dim": input_dim,
            "onnx": str(onnx_path), "onnx_sha256": sha256_file(onnx_path),
        }
        if not a.onnx_only:
            eng, cmd = build_reranker_engine_with_trtexec(
                onnx_path, engine_dir / "pair_reranker.plan", input_dim,
                precision=a.precision, workspace_mib=min(a.workspace_mib, 1024), trtexec=a.trtexec,
            )
            rec.update({"engine": str(eng), "engine_sha256": sha256_file(eng), "engine_build": cmd})
        manifest["models"]["pair_reranker"] = rec
    elif not a.skip_reranker:
        manifest["notes"].append("No reranker checkpoint was found; TensorRT pair-reranker export was skipped.")

    write_json(out / "export_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n[DONE] TensorRT export manifest: {out / 'export_manifest.json'}")


if __name__ == "__main__":
    main()
