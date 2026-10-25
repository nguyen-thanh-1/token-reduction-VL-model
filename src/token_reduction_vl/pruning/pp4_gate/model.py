"""PP4 gate network: one light transformer block producing ``log α`` per token (``method.md`` §2)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .features import POSITION_DIM, GateFeatures
from .hard_concrete import initial_log_alpha

HIDDEN = 256
LEVEL_DIM = 64
UNIT_DIM = HIDDEN + 3 * LEVEL_DIM + 2 + POSITION_DIM  # 466


class GateNet(nn.Module):
    def __init__(self, input_dim: int = 2048, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj_v = nn.Linear(input_dim, HIDDEN)
        self.proj_levels = nn.ModuleList([nn.Linear(input_dim, LEVEL_DIM) for _ in range(3)])
        self.proj_q = nn.Linear(input_dim, HIDDEN)
        self.inp = nn.Linear(UNIT_DIM, HIDDEN)
        self.norm_self = nn.LayerNorm(HIDDEN)
        self.self_attn = nn.MultiheadAttention(HIDDEN, heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(HIDDEN)
        self.cross_attn = nn.MultiheadAttention(HIDDEN, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(HIDDEN)
        self.ffn = nn.Sequential(nn.Linear(HIDDEN, 1024), nn.GELU(), nn.Linear(1024, HIDDEN))
        self.norm_out = nn.LayerNorm(HIDDEN)
        self.out = nn.Linear(HIDDEN, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, initial_log_alpha())  # π ≈ 0.5 for every token at start

    def forward(self, x: GateFeatures) -> Tensor:
        """``log α`` of shape ``[N]``."""

        levels = [proj(level) for proj, level in zip(self.proj_levels, x.f, strict=True)]
        u = torch.cat([self.proj_v(x.v), *levels, torch.stack([x.z_rel, x.z_ms], dim=1), x.positions], dim=1)
        h = self.inp(u).unsqueeze(0)
        normed = self.norm_self(h)
        h = h + self.self_attn(normed, normed, normed, need_weights=False)[0]
        if x.q.shape[0]:
            q = self.proj_q(x.q).unsqueeze(0)
            h = h + self.cross_attn(self.norm_cross(h), q, q, need_weights=False)[0]
        h = h + self.ffn(self.norm_ffn(h))
        return self.out(self.norm_out(h)).reshape(-1)
