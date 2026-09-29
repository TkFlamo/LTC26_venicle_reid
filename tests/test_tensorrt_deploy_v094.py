from pathlib import Path

import torch
import torch.nn.functional as F

from vehicle_fingerprint.tensorrt_deploy import (
    discover_v094_global_checkpoints,
    summarize_times,
    weighted_ensemble_embedding,
    _torch_dtype_for_precision,
)


def test_weighted_ensemble_embedding_is_score_equivalent():
    torch.manual_seed(4)
    qv=F.normalize(torch.randn(3,7),dim=-1); gv=F.normalize(torch.randn(5,7),dim=-1)
    qc=F.normalize(torch.randn(3,11),dim=-1); gc=F.normalize(torch.randn(5,11),dim=-1)
    w=.45
    q=weighted_ensemble_embedding(qv,qc,w); g=weighted_ensemble_embedding(gv,gc,w)
    got=q@g.T
    ref=(1-w)*(qv@gv.T)+w*(qc@gc.T)
    assert torch.allclose(got,ref,atol=2e-6,rtol=2e-6)


def test_checkpoint_discovery_prefers_direct_v094(tmp_path: Path):
    rr=tmp_path/'runs/full_single_v094_v5exact_base_veri'
    v=rr/'global_v5_exact_direct/vit_base/384x576/project_best.pt'
    c=rr/'global_v5_exact_direct/convnext_base/384x576/project_best.pt'
    v.parent.mkdir(parents=True);c.parent.mkdir(parents=True)
    v.write_bytes(b'vit');c.write_bytes(b'cn')
    got=discover_v094_global_checkpoints(tmp_path)
    assert got['vit']==v
    assert got['convnext']==c


def test_summarize_times():
    s=summarize_times([1.,2.,3.,4.])
    assert s['n']==4
    assert s['mean_ms']==2.5
    assert s['p95_ms']>=3.0


def test_trt11_has_no_legacy_fp16_flag():
    from vehicle_fingerprint.tensorrt_deploy import _builder_precision_flag
    class Flags:
        DEBUG = object()
    class FakeTRT:
        BuilderFlag = Flags
        __version__ = "11.3.0.99"
    assert _builder_precision_flag(FakeTRT, "fp16") is None
    assert _builder_precision_flag(FakeTRT, "bf16") is None


def test_trt10_uses_legacy_fp16_flag():
    from vehicle_fingerprint.tensorrt_deploy import _builder_precision_flag
    sentinel = object()
    class Flags:
        FP16 = sentinel
    class FakeTRT:
        BuilderFlag = Flags
        __version__ = "10.13.3"
    assert _builder_precision_flag(FakeTRT, "fp16") is sentinel


def test_precision_to_torch_dtype():
    assert _torch_dtype_for_precision("fp32") is torch.float32
    assert _torch_dtype_for_precision("fp16") is torch.float16
    assert _torch_dtype_for_precision("bf16") is torch.bfloat16


def test_precision_to_torch_dtype_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        _torch_dtype_for_precision("tf32")


def test_checkpoint_discovery_explicit_run_root_wins(tmp_path: Path):
    old=tmp_path/'runs/full_single_v094_v5exact_base/global_v5_exact/vit_base/384x576/project_best.pt'
    cur=tmp_path/'runs/full_single_v094_v5exact_base_veri/global_v5_exact_direct/vit_base/384x576/project_best.pt'
    old.parent.mkdir(parents=True); old.write_bytes(b'old')
    cur.parent.mkdir(parents=True); cur.write_bytes(b'current')
    got=discover_v094_global_checkpoints(tmp_path, 'runs/full_single_v094_v5exact_base_veri')
    assert got['vit']==cur


def test_trt11_set_shape_none_is_success():
    from vehicle_fingerprint.tensorrt_deploy import _set_profile_shape_compat
    class Profile:
        def set_shape(self, name, mn, op, mx):
            assert name == "input"
            assert mn == (1,3,384,576)
            assert op == (8,3,384,576)
            assert mx == (32,3,384,576)
            return None  # TensorRT 11.3 documented success return
    assert _set_profile_shape_compat(Profile(), "input", (1,3,384,576), (8,3,384,576), (32,3,384,576)) is None


def test_legacy_set_shape_false_is_failure():
    import pytest
    from vehicle_fingerprint.tensorrt_deploy import _set_profile_shape_compat
    class Profile:
        def set_shape(self, *args):
            return False
    with pytest.raises(RuntimeError, match="rejected optimization profile"):
        _set_profile_shape_compat(Profile(), "input", (1,3,384,576), (8,3,384,576), (32,3,384,576))


def test_profile_validation_allows_dynamic_batch_and_fixed_hw():
    from vehicle_fingerprint.tensorrt_deploy import _validate_profile_shapes
    _validate_profile_shapes((-1,3,384,576), (1,3,384,576), (8,3,384,576), (32,3,384,576), input_name="input")


def test_profile_validation_rejects_static_batch_with_dynamic_profile():
    import pytest
    from vehicle_fingerprint.tensorrt_deploy import _validate_profile_shapes
    with pytest.raises(RuntimeError, match="dimension 0 is static"):
        _validate_profile_shapes((1,3,384,576), (1,3,384,576), (8,3,384,576), (32,3,384,576), input_name="input")


def test_full_wrapper_fixed_vit_pool_matches_adaptive_exactly():
    from torch import nn
    from vehicle_fingerprint.tensorrt_deploy import FullFeatureWrapper
    class BB:
        is_vit = True
        patch_size = (16, 16)
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__(); self.backbone = BB(); self.local_grid = (4, 6)
    w = FullFeatureWrapper(Dummy(), (384, 576))
    assert w._fixed_pool_kernel == (6, 6)
    x = torch.randn(2, 160, 24, 36)
    assert torch.allclose(w._local_pool(x), F.adaptive_avg_pool2d(x, (4, 6)), atol=1e-7, rtol=1e-6)


def test_full_wrapper_nondivisible_map_keeps_adaptive_fallback():
    from torch import nn
    from vehicle_fingerprint.tensorrt_deploy import FullFeatureWrapper
    class BB:
        is_vit = True
        patch_size = (16, 16)
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__(); self.backbone = BB(); self.local_grid = (5, 7)
    w = FullFeatureWrapper(Dummy(), (384, 576))
    assert w._fixed_pool_kernel is None
    x = torch.randn(1, 8, 24, 36)
    assert torch.allclose(w._local_pool(x), F.adaptive_avg_pool2d(x, (5, 7)))
