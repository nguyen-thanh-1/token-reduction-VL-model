"""Shared, method-agnostic pruning interface (see ``00-common/README.md`` §4.0).

Every candidate method implements :class:`VisualTokenPruner`. The inference
wrapper only calls ``select`` and then rebuilds the LLM inputs, so all methods
share the same inference and evaluation code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PrunerInput:
    """Everything a pruner may look at for one image-question pair."""

    v: Tensor
    """Main visual embeddings from the final PatchMerger, ``[N, 2048]``."""
    deepstack: list[Tensor]
    """DeepStack features ``f1, f2, f3``, each ``[N, 2048]``."""
    grid_hw: tuple[int, int]
    """Merged token grid ``(H', W')``; token ``j`` sits at ``(j // W', j % W')``."""
    question_ids: Tensor
    """Token ids of the question text plus option texts, ``[N_q]`` (PP1 §3.1)."""
    question_tokens: tuple[str, ...]
    """Decoded string of each id in ``question_ids`` (used for stopword filtering)."""
    embed_tokens: nn.Embedding
    """The LLM input embedding table (read-only)."""
    ratio: float | None
    """Fixed keep ratio, or ``None`` when the method chooses ``K`` itself."""

    @property
    def num_tokens(self) -> int:
        return int(self.v.shape[0])


@dataclass(frozen=True)
class PrunerOutput:
    keep_indices: Tensor
    """``[K]`` int64, strictly increasing, ``1 <= K <= N``."""
    v_kept: Tensor
    """``[K, 2048]`` embeddings scattered into the image positions."""
    deepstack_kept: list[Tensor]
    """Three ``[K, 2048]`` tensors added after decoder layers 0/1/2."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Must contain ``K``, ``N``, ``method`` and ``ratio_target``."""


REQUIRED_METADATA = ("K", "N", "method", "ratio_target")

BUDGET_LEVELS: dict[str, float] = {
    "L1": 1 / 18,
    "L2": 1 / 9,
    "L3": 2 / 9,
    "L4": 1 / 3,
    "L5": 1 / 2,
    "L6": 3 / 4,
    "L7": 1.0,
}
"""The seven shared keep ratios (``00-common/README.md`` §6)."""


class VisualTokenPruner(Protocol):
    name: str

    def select(self, x: PrunerInput) -> PrunerOutput: ...


def budget_for_ratio(ratio: float, num_tokens: int) -> int:
    """``K = clip(ceil(r * N), 1, N)``."""

    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"Keep ratio must be in (0, 1], got {ratio}.")
    # The tolerance keeps exact levels such as 1/3 * 384 = 128 from rounding up to 129.
    return max(1, min(num_tokens, math.ceil(ratio * num_tokens - 1e-6)))


def validate_output(x: PrunerInput, out: PrunerOutput) -> None:
    """Raise ``ValueError`` when a pruner breaks the shared contract."""

    keep = out.keep_indices
    n = x.num_tokens
    if keep.ndim != 1 or keep.dtype != torch.long:
        raise ValueError("keep_indices must be a 1-D int64 tensor.")
    k = int(keep.numel())
    if not 1 <= k <= n:
        raise ValueError(f"Expected 1 <= K <= N, got K={k}, N={n}.")
    if k > 1 and not bool((keep[1:] > keep[:-1]).all()):
        raise ValueError("keep_indices must be strictly increasing.")
    if int(keep.min()) < 0 or int(keep.max()) >= n:
        raise ValueError("keep_indices out of range.")
    if out.v_kept.shape != (k, x.v.shape[1]):
        raise ValueError(f"v_kept must be [{k}, {x.v.shape[1]}], got {tuple(out.v_kept.shape)}.")
    if len(out.deepstack_kept) != len(x.deepstack):
        raise ValueError("deepstack_kept must have one tensor per DeepStack level.")
    for level, feature in enumerate(out.deepstack_kept):
        if feature.shape != (k, x.deepstack[level].shape[1]):
            raise ValueError(f"deepstack_kept[{level}] must be [{k}, ...], got {tuple(feature.shape)}.")
    missing = [key for key in REQUIRED_METADATA if key not in out.metadata]
    if missing:
        raise ValueError(f"metadata is missing {missing}.")


class NoOpPruner:
    """Reference implementation representing the full-token baseline."""

    name = "none"

    def select(self, x: PrunerInput) -> PrunerOutput:
        n = x.num_tokens
        return PrunerOutput(
            keep_indices=torch.arange(n, dtype=torch.long, device=x.v.device),
            v_kept=x.v,
            deepstack_kept=list(x.deepstack),
            metadata={"K": n, "N": n, "method": self.name, "ratio_target": 1.0},
        )
