from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """ArcFace head matching the supplied V4 baseline implementation."""
    def __init__(self, in_features: int, out_features: int, scale: float = 30.0, margin: float = 0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale); self.margin = float(margin)
        self.cm = math.cos(self.margin); self.sm = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin); self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=1e-7))
        phi = cosine * self.cm - sine * self.sm
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels[:, None], 1.0)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale


class ResidualPartBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        groups = 8 if dim % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, dim)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.gelu(self.norm1(x)))
        y = self.conv2(F.gelu(self.norm2(y)))
        return x + y


class DinoPartHead(nn.Module):
    """Lightweight multi-label semantic decoder on the native DINO patch grid.

    This is intentionally not a high-resolution segmentation network.  Its job is to estimate
    semantic support for DINO patches, exactly at the granularity used by part-token pooling.
    """

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256, blocks: int = 2):
        super().__init__()
        groups = 8 if hidden_dim % 8 == 0 else 1
        layers: list[nn.Module] = [
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        ]
        layers += [ResidualPartBlock(hidden_dim) for _ in range(int(blocks))]
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(hidden_dim, num_classes, 1)
        # Start from a sparse semantic prior instead of p=0.5 everywhere. This prevents
        # untrained/rare part channels from appearing visible before Carparts supervision is learned.
        nn.init.constant_(self.out.bias, -2.2)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(fmap))


class SemanticPartAttention(nn.Module):
    """Learned part queries with soft semantic guidance from the Carparts-Seg-trained DINO part head.

    The semantic probability acts as a prior, not a hard mask. This lets each query focus on
    identity-relevant subregions while keeping the semantic meaning learned from Carparts-Seg.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_parts: int,
        *,
        heads: int = 8,
        prior_strength: float = 2.5,
        visibility_threshold: float = 0.28,
        visibility_topk: int = 3,
        detach_semantic_prior: bool = True,
    ):
        super().__init__()
        if out_dim % heads:
            raise ValueError(f"out_dim={out_dim} must be divisible by heads={heads}")
        self.num_parts = int(num_parts)
        self.out_dim = int(out_dim)
        self.heads = int(heads)
        self.head_dim = out_dim // heads
        self.prior_strength = float(prior_strength)
        self.visibility_threshold = float(visibility_threshold)
        self.visibility_topk = int(visibility_topk)
        self.detach_semantic_prior = bool(detach_semantic_prior)
        self.query = nn.Parameter(torch.randn(num_parts, out_dim) * 0.02)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)
        self.out = nn.Sequential(nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(
        self,
        fmap: torch.Tensor,
        semantic_probs: torch.Tensor,
        *,
        foreground_prob: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, H, W = fmap.shape
        x = fmap.flatten(2).transpose(1, 2)  # B,N,D
        k = self.key(x).view(B, H * W, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).view(B, H * W, self.heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.query.view(self.num_parts, self.heads, self.head_dim).permute(1, 0, 2)  # h,P,d
        score = torch.einsum("hpd,bhnd->bhpn", q, k) / math.sqrt(self.head_dim)

        prior = semantic_probs.flatten(2)  # B,P,N
        if foreground_prob is not None:
            prior = prior * foreground_prob.flatten(2).clamp(0, 1)
        if self.detach_semantic_prior:
            prior = prior.detach()
        score = score + self.prior_strength * torch.log(prior[:, None].clamp_min(1e-4))
        attn = torch.softmax(score, dim=-1)
        token = torch.einsum("bhpn,bhnd->bhpd", attn, v).permute(0, 2, 1, 3).reshape(B, self.num_parts, self.out_dim)
        token = self.out(token)

        flat = semantic_probs.flatten(2)
        kvis = min(max(1, self.visibility_topk), flat.shape[-1])
        vis_score = flat.topk(kvis, dim=-1).values.mean(dim=-1)
        visibility = vis_score >= self.visibility_threshold
        return token, visibility, vis_score

    def pool_with_mask(self, fmap: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Oracle masked pooling in the same value space, used only to bootstrap part queries."""
        B, D, H, W = fmap.shape
        if masks.shape[-2:] != (H, W):
            masks = F.interpolate(masks.float(), size=(H, W), mode="area")
        else:
            masks = masks.float()
        x = fmap.flatten(2).transpose(1, 2)
        value = self.value(x)  # B,N,E
        m = masks.flatten(2).clamp_min(0.0)
        token = torch.einsum("bpn,bne->bpe", m, value) / m.sum(dim=-1).clamp_min(1e-6)[..., None]
        return self.out(token)


class PartTokenFusion(nn.Module):
    """Fuse semantic part tokens as a *bounded residual* on a strong global embedding.

    The previous implementation used an unconstrained scalar gate.  A noisy part branch could
    therefore dominate the identity representation and erase a strong triplet-only baseline.
    Here the part contribution is explicitly bounded and starts almost closed.  Validation-time
    score fusion can always fall back to the untouched global branch.
    """

    def __init__(
        self, dim: int, num_parts: int, layers: int = 2, heads: int = 8, dropout: float = 0.1,
        max_residual: float = 0.35, initial_gate: float = 0.02,
    ):
        super().__init__()
        self.num_parts = num_parts
        self.max_residual = float(max_residual)
        self.type_embed = nn.Parameter(torch.randn(1, num_parts + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        # Parameterise a bounded gate through sigmoid.  initial_gate is expressed in the final
        # residual scale (e.g. 0.02 means only a 2% residual at startup).
        frac = min(max(float(initial_gate) / max(self.max_residual, 1e-6), 1e-4), 1.0 - 1e-4)
        init_logit = math.log(frac / (1.0 - frac))
        self.part_gate_logit = nn.Parameter(torch.tensor([init_logit], dtype=torch.float32))

    @property
    def gate(self) -> torch.Tensor:
        return self.max_residual * torch.sigmoid(self.part_gate_logit)

    def forward(self, global_token: torch.Tensor, part_tokens: torch.Tensor, visibility: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.cat([global_token[:, None, :], part_tokens], dim=1)
        tokens = tokens + self.type_embed[:, : tokens.shape[1]]
        pad = torch.cat([
            torch.zeros((tokens.shape[0], 1), device=tokens.device, dtype=torch.bool),
            ~visibility.bool(),
        ], dim=1)
        y = self.encoder(tokens, src_key_padding_mask=pad)
        y = self.norm(y)
        fused_global = global_token + self.gate.to(global_token.dtype) * (y[:, 0] - global_token)
        return fused_global, y[:, 1:]


class MultiScaleSpatialFusion(nn.Module):
    """Fuse several ConvNeXt stages / ViT depths into one detail map.

    Each source receives an independent 1x1 projection and a learnable scalar reliability. Maps are
    resized to the highest spatial resolution and combined with softmax-normalised scale weights.
    This preserves fine boundaries from shallow features while retaining semantics from deep ones.
    """
    def __init__(self, in_channels: list[int], out_dim: int = 256):
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels must not be empty")
        groups = 8 if out_dim % 8 == 0 else 1
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(int(c), out_dim, 1, bias=False), nn.GroupNorm(groups, out_dim), nn.GELU())
            for c in in_channels
        ])
        self.scale_logits = nn.Parameter(torch.zeros(len(in_channels), dtype=torch.float32))
        self.refine = nn.Sequential(ResidualPartBlock(out_dim), ResidualPartBlock(out_dim))
        self.out_dim = int(out_dim)

    def forward(self, maps: list[torch.Tensor]) -> torch.Tensor:
        if len(maps) != len(self.proj):
            if len(maps) == 1 and len(self.proj) > 1:
                # Backward-compatible checkpoint/data path: use the deepest projection only.
                return self.refine(self.proj[-1](maps[0]))
            raise RuntimeError(f"Expected {len(self.proj)} feature maps, got {len(maps)}")
        target_h = max(int(x.shape[-2]) for x in maps)
        target_w = max(int(x.shape[-1]) for x in maps)
        ys=[]
        for p,x in zip(self.proj,maps):
            y=p(x)
            if y.shape[-2:] != (target_h,target_w):
                y=F.interpolate(y,size=(target_h,target_w),mode="bilinear",align_corners=False)
            ys.append(y)
        w=torch.softmax(self.scale_logits,dim=0).to(ys[0].dtype)
        fused=sum(wi*yi for wi,yi in zip(w,ys))
        return self.refine(fused)
