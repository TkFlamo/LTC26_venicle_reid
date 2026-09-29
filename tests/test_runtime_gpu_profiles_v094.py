from __future__ import annotations

import torch
from torch import nn

from vehicle_fingerprint.runtime_pt import _concat_output_chunks, forward_loaded_models


class Tiny(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, x):
        y = x.mean(dim=(2, 3)) * self.scale
        return {"z_global": y, "z_fused": y + 1, "z_local": y - 1, "parts": y[:, None],
                "visibility": torch.ones((x.shape[0], 1), dtype=torch.bool, device=x.device),
                "visibility_score": torch.ones((x.shape[0], 1), device=x.device),
                "local": y[:, None], "logits": None, "part_logits": y[:, :, None, None], "part_probs": y[:, :, None, None]}


def test_concat_chunks_preserves_batch_order():
    a = {"x": torch.tensor([[1.0], [2.0]]), "none": None}
    b = {"x": torch.tensor([[3.0]]), "none": None}
    out = _concat_output_chunks([a, b])
    assert out["x"].flatten().tolist() == [1.0, 2.0, 3.0]
    assert out["none"] is None


def test_parallel_helper_cpu_falls_back_to_exact_sequential():
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    models = [{"model": Tiny(1.0)}, {"model": Tiny(2.0)}]
    seq = forward_loaded_models(models, x, torch.device("cpu"), precision="fp32", parallel_mode="sequential")
    streams = forward_loaded_models(models, x, torch.device("cpu"), precision="fp32", parallel_mode="streams", stream_microbatch=1)
    for a, b in zip(seq, streams):
        assert torch.equal(a["z_global"], b["z_global"])
        assert torch.equal(a["z_fused"], b["z_fused"])
