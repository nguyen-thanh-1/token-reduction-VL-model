"""PP3 score heads: 3a MLP, 3b CNN with FiLM, 3c light transformer (``method.md`` §3).

Each head maps one sample's :class:`TokenFeatures` to raw scores ``s`` of
shape ``[N]``; the pruner uses ``q̂ = softplus(s) + 1e-3`` inside the DPP kernel.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .features import POSITION_DIM, TokenFeatures

TOKEN_DIM = 256
LEVEL_DIM = 64
QUESTION_DIM = 256
UNIT_DIM = TOKEN_DIM + 3 * LEVEL_DIM + 2 + POSITION_DIM + 1  # 467


class TokenEncoder(nn.Module):
    """Builds ``u_i`` (467-d) and the question summary ``x_q`` (256-d)."""

    def __init__(self, v_dim: int, level_dim: int, q_dim: int) -> None:
        super().__init__()
        self.proj_v = nn.Linear(v_dim, TOKEN_DIM)
        self.proj_levels = nn.ModuleList([nn.Linear(level_dim, LEVEL_DIM) for _ in range(3)])
        self.proj_q = nn.Linear(q_dim, QUESTION_DIM)

    def forward(self, x: TokenFeatures) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pv = self.proj_v(x.v)
        levels = [proj(level) for proj, level in zip(self.proj_levels, x.f, strict=True)]
        n = x.num_tokens
        scalars = torch.stack([x.z_rel, x.z_ms], dim=1)
        log_r = torch.full((n, 1), x.log_ratio, device=pv.device)
        u = torch.cat([pv, *levels, scalars, x.positions, log_r], dim=1)
        q_tokens = self.proj_q(x.q) if x.q.shape[0] else torch.zeros((0, QUESTION_DIM), device=pv.device)
        q_mean = q_tokens.mean(dim=0) if q_tokens.shape[0] else torch.zeros(QUESTION_DIM, device=pv.device)
        return u, pv, q_mean, q_tokens


class ScoreMLP(nn.Module):
    """3a: per-token MLP on ``[u_i ; x_q ; P_v(v_i) ⊙ x_q]`` (979-d)."""

    def __init__(self, v_dim: int, level_dim: int, q_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = TokenEncoder(v_dim, level_dim, q_dim)
        self.mlp = nn.Sequential(
            nn.Linear(UNIT_DIM + 2 * QUESTION_DIM, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: TokenFeatures) -> Tensor:
        u, pv, q_mean, _ = self.encoder(x)
        q = q_mean.expand(x.num_tokens, -1)
        return self.mlp(torch.cat([u, q, pv * q], dim=1)).squeeze(-1)


class ScoreCNN(nn.Module):
    """3b: three residual 3×3 convolutions on the ``H'×W'`` grid, FiLM-conditioned on the question."""

    def __init__(self, v_dim: int, level_dim: int, q_dim: int, blocks: int = 3) -> None:
        super().__init__()
        self.encoder = TokenEncoder(v_dim, level_dim, q_dim)
        self.inp = nn.Linear(UNIT_DIM, TOKEN_DIM)
        self.film = nn.Linear(QUESTION_DIM, 2 * TOKEN_DIM)
        self.convs = nn.ModuleList([nn.Conv2d(TOKEN_DIM, TOKEN_DIM, 3, padding=1) for _ in range(blocks)])
        self.out = nn.Conv2d(TOKEN_DIM, 1, 1)
        self.act = nn.GELU()

    def forward(self, x: TokenFeatures) -> Tensor:
        u, _, q_mean, _ = self.encoder(x)
        height, width = x.grid_hw
        h = self.inp(u).T.reshape(1, TOKEN_DIM, height, width)
        gamma, beta = self.film(q_mean).view(2, TOKEN_DIM, 1, 1)
        for conv in self.convs:
            h = h + self.act(gamma * conv(h) + beta)
        return self.out(h).reshape(-1)


class ScoreTransformer(nn.Module):
    """3c: one block with token self-attention and cross-attention to question tokens."""

    def __init__(self, v_dim: int, level_dim: int, q_dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = TokenEncoder(v_dim, level_dim, q_dim)
        self.inp = nn.Linear(UNIT_DIM, TOKEN_DIM)
        self.norm_self = nn.LayerNorm(TOKEN_DIM)
        self.self_attn = nn.MultiheadAttention(TOKEN_DIM, heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(TOKEN_DIM)
        self.cross_attn = nn.MultiheadAttention(TOKEN_DIM, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(TOKEN_DIM)
        self.ffn = nn.Sequential(nn.Linear(TOKEN_DIM, 1024), nn.GELU(), nn.Linear(1024, TOKEN_DIM))
        self.norm_out = nn.LayerNorm(TOKEN_DIM)
        self.out = nn.Linear(TOKEN_DIM, 1)

    def forward(self, x: TokenFeatures) -> Tensor:
        u, _, _, q_tokens = self.encoder(x)
        h = self.inp(u).unsqueeze(0)
        normed = self.norm_self(h)
        h = h + self.self_attn(normed, normed, normed, need_weights=False)[0]
        if q_tokens.shape[0]:
            h = h + self.cross_attn(self.norm_cross(h), q_tokens.unsqueeze(0), q_tokens.unsqueeze(0), need_weights=False)[0]
        h = h + self.ffn(self.norm_ffn(h))
        return self.out(self.norm_out(h)).reshape(-1)


ARCHITECTURES: dict[str, type[nn.Module]] = {"3a": ScoreMLP, "3b": ScoreCNN, "3c": ScoreTransformer}


def build_score_head(arch: str, dims: tuple[int, int, int]) -> nn.Module:
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown score-head architecture {arch!r}; choose from {sorted(ARCHITECTURES)}.")
    return ARCHITECTURES[arch](*dims)


def quality_from_scores(scores: Tensor) -> Tensor:
    """``q̂ = softplus(s) + 1e-3`` — strictly positive, as the DPP kernel requires."""

    return nn.functional.softplus(scores) + 1e-3
