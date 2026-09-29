from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def local_match_score(q: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Symmetric nearest-neighbour evidence for [B,L,D] descriptors."""
    sim = torch.einsum("bld,bmd->blm", q, g)
    q2g = sim.max(dim=2).values.mean(dim=1)
    g2q = sim.max(dim=1).values.mean(dim=1)
    return 0.5 * (q2g + g2q)


def pair_features(q: dict[str, torch.Tensor], g: dict[str, torch.Tensor]) -> torch.Tensor:
    glob = (q["z_fused"] * g["z_fused"]).sum(-1, keepdim=True)
    rawg = (q["z_global"] * g["z_global"]).sum(-1, keepdim=True)
    ps = (q["parts"] * g["parts"]).sum(-1)
    joint = q["visibility"].bool() & g["visibility"].bool()
    ps_masked = torch.where(joint, ps, torch.zeros_like(ps))
    vis = joint.float()
    local_global = ((q["z_local"] * g["z_local"]).sum(-1, keepdim=True)
                    if "z_local" in q and "z_local" in g else torch.zeros_like(glob))
    local = local_match_score(q["local"], g["local"])[:, None]
    # [global fused, raw global, pooled-local, patch-local, part scores, visibility flags]
    return torch.cat([glob, rawg, local_global, local, ps_masked, vis], dim=1)


class PairReranker(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
