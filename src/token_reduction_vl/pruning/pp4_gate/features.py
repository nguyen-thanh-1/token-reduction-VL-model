"""Per-token inputs of the PP4 gate network (``method.md`` §2), built online."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from token_reduction_vl.pruning.base import PrunerInput

from .dsfold_base import ScoreBundle, unit_rows
from .dsfold_base.scoring import question_embeddings

FOURIER_FREQUENCIES = 4
POSITION_DIM = 4 * FOURIER_FREQUENCIES


def fourier_positions(num_tokens: int, grid_hw: tuple[int, int], device: torch.device) -> Tensor:
    """``[N, 16]``: ``sin/cos(2^k·π·x)`` of normalized row and column, k = 0..3."""

    height, width = grid_hw
    index = torch.arange(num_tokens, device=device)
    row = torch.div(index, width, rounding_mode="floor").float() / max(height, 1)
    col = (index % width).float() / max(width, 1)
    parts = []
    for k in range(FOURIER_FREQUENCIES):
        scale = (2**k) * math.pi
        parts.extend([torch.sin(scale * row), torch.cos(scale * row), torch.sin(scale * col), torch.cos(scale * col)])
    return torch.stack(parts, dim=1)


@dataclass(frozen=True)
class GateFeatures:
    v: Tensor  # [N, 2048] unit rows
    f: list[Tensor]  # 3 × [N, 2048] standardized DeepStack
    z_rel: Tensor  # [N]
    z_ms: Tensor  # [N]
    positions: Tensor  # [N, 16]
    q: Tensor  # [N_q, 2048] unit rows; N_q may be 0

    @property
    def num_tokens(self) -> int:
        return int(self.v.shape[0])


def build_gate_features(x: PrunerInput, scores: ScoreBundle, *, stopwords: tuple[str, ...]) -> GateFeatures:
    q = question_embeddings(x, stopwords)
    q = unit_rows(q.to(x.v.device)) if q is not None else torch.zeros((0, x.v.shape[1]), device=x.v.device)
    return GateFeatures(
        v=scores.v_unit,
        f=list(scores.f_hat),
        z_rel=scores.z_rel,
        z_ms=scores.z_ms,
        positions=fourier_positions(x.num_tokens, x.grid_hw, x.v.device),
        q=q,
    )
