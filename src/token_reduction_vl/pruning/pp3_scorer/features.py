"""Per-token inputs of the PP3 score head (``pp3-dsfold-scorer/method.md`` §2).

Route B trains offline from stored features. Raw ``v`` and DeepStack features
for 50k samples would need roughly 220 GB, so a *fixed* PCA (fitted once,
never trained) compresses them before storage; the same PCA runs at inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
class TokenFeatures:
    v: Tensor  # [N, Dv] unit-normalized main stream (optionally PCA-compressed)
    f: list[Tensor]  # 3 × [N, Df] standardized DeepStack (optionally PCA-compressed)
    z_rel: Tensor  # [N]
    z_ms: Tensor  # [N]
    positions: Tensor  # [N, 16]
    q: Tensor  # [N_q, Dq] question token embeddings (unit rows); N_q may be 0
    grid_hw: tuple[int, int]
    log_ratio: float

    @property
    def num_tokens(self) -> int:
        return int(self.v.shape[0])

    def to(self, device: torch.device) -> TokenFeatures:
        return TokenFeatures(
            v=self.v.to(device),
            f=[level.to(device) for level in self.f],
            z_rel=self.z_rel.to(device),
            z_ms=self.z_ms.to(device),
            positions=self.positions.to(device),
            q=self.q.to(device),
            grid_hw=self.grid_hw,
            log_ratio=self.log_ratio,
        )


class PCAProjection:
    """Fixed linear map ``x ↦ (x − mean) @ components``; not trainable."""

    def __init__(self, mean: Tensor, components: Tensor) -> None:
        self.mean = mean.float()
        self.components = components.float()  # [D_in, D_out]

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[1])

    @classmethod
    def fit(cls, samples: Tensor, output_dim: int) -> PCAProjection:
        samples = samples.float()
        mean = samples.mean(dim=0)
        _, _, vh = torch.linalg.svd(samples - mean, full_matrices=False)
        return cls(mean, vh[:output_dim].T.contiguous())

    def __call__(self, x: Tensor) -> Tensor:
        return (x.float() - self.mean.to(x.device)) @ self.components.to(x.device)


class FeatureCompressor:
    """PCA for ``v`` (→256), each DeepStack level (→64) and question tokens (→256)."""

    def __init__(self, v: PCAProjection, levels: list[PCAProjection], q: PCAProjection) -> None:
        self.v = v
        self.levels = levels
        self.q = q

    @property
    def dims(self) -> tuple[int, int, int]:
        return self.v.output_dim, self.levels[0].output_dim, self.q.output_dim

    @classmethod
    def fit(
        cls,
        v_rows: Tensor,
        level_rows: list[Tensor],
        q_rows: Tensor,
        *,
        v_dim: int = 256,
        level_dim: int = 64,
        q_dim: int = 256,
    ) -> FeatureCompressor:
        return cls(
            PCAProjection.fit(v_rows, v_dim),
            [PCAProjection.fit(rows, level_dim) for rows in level_rows],
            PCAProjection.fit(q_rows, q_dim),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"v_mean": self.v.mean, "v_comp": self.v.components, "q_mean": self.q.mean, "q_comp": self.q.components}
        for index, level in enumerate(self.levels):
            arrays[f"f{index}_mean"] = level.mean
            arrays[f"f{index}_comp"] = level.components
        np.savez(path, **{key: value.cpu().numpy() for key, value in arrays.items()})

    @classmethod
    def load(cls, path: Path) -> FeatureCompressor:
        data = np.load(path)
        tensor = lambda key: torch.from_numpy(data[key])  # noqa: E731
        levels = [PCAProjection(tensor(f"f{i}_mean"), tensor(f"f{i}_comp")) for i in range(3)]
        return cls(PCAProjection(tensor("v_mean"), tensor("v_comp")), levels, PCAProjection(tensor("q_mean"), tensor("q_comp")))


def question_rows(x: PrunerInput, stopwords: tuple[str, ...]) -> Tensor:
    """Unit-norm question token embeddings ``[N_q, D]`` (``N_q`` may be 0)."""

    q = question_embeddings(x, stopwords)
    if q is None:
        return torch.zeros((0, x.v.shape[1]), device=x.v.device)
    return unit_rows(q.to(x.v.device))


def build_token_features(
    x: PrunerInput,
    scores: ScoreBundle,
    *,
    stopwords: tuple[str, ...],
    ratio: float,
    compressor: FeatureCompressor | None = None,
) -> TokenFeatures:
    """Assemble ``TokenFeatures`` from a pruner input and its DSFold S1 scores."""

    q = question_rows(x, stopwords)
    v = scores.v_unit
    f = scores.f_hat
    if compressor is not None:
        v = compressor.v(v)
        f = [projection(level) for projection, level in zip(compressor.levels, f, strict=True)]
        q = compressor.q(q) if q.numel() else torch.zeros((0, compressor.q.output_dim), device=v.device)
    return TokenFeatures(
        v=v,
        f=list(f),
        z_rel=scores.z_rel,
        z_ms=scores.z_ms,
        positions=fourier_positions(x.num_tokens, x.grid_hw, v.device),
        q=q,
        grid_hw=x.grid_hw,
        log_ratio=math.log(ratio),
    )
