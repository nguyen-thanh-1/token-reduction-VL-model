"""PP3 focus + context selection (ADR 0004).

The budget ``K`` is split in two:

* **focus** — regions grown from the focus head's scores (trained on
  question-specific, contrastive attention). Each step adds the free token with
  the highest standardised score plus ``beta`` if it touches the focus set
  (8-neighbourhood), so regions stay connected unless another peak is clearly
  higher.
* **context** — either one token per coarse grid cell, the best by the global
  head's quality, so the whole image stays covered (``context_dpp = 0``), or
  DSFold's DPP on the global quality over the remaining tokens, i.e. the
  3d_attn50k selection minus what focus took (``context_dpp = 1``), which keeps
  the image-level tokens the LLM relies on.

The focus share grows with how peaked the focus distribution is: a clear
target gets most of the budget, a flat map (nothing matches) falls back to
coverage. Dropped tokens are folded as in DSFold (S3/S4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from token_reduction_vl.pruning.base import PrunerInput, PrunerOutput, budget_for_ratio

from .dsfold_base import DSFoldParams, load_dsfold_params
from .dsfold_base.dpp import select_order
from .features import FeatureCompressor, build_token_features
from .pruner import ScoredDSFoldPruner
from .train import load_score_head


@dataclass(frozen=True)
class FocusContextParams:
    beta: float = 1.0
    """Bonus (in standard deviations of the focus score) for tokens next to the focus set."""
    share_min: float = 0.3
    """Focus share of the budget when the focus map is flat."""
    share_max: float = 0.8
    """Focus share when the focus map is peaked."""
    peak_top: float = 0.1
    """Peakedness = focus probability mass on the top ``peak_top`` fraction of tokens."""
    context_dpp: float = 0.0
    """1 = context by DPP on the global quality; 0 = one token per coarse grid cell."""

    def with_overrides(self, **overrides: Any) -> FocusContextParams:
        known = {field.name for field in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown focus-context parameters {sorted(unknown)}; choose from {sorted(known)}.")
        return replace(self, **{key: float(value) for key, value in overrides.items()})


_NEIGHBORS: dict[tuple[int, int], Tensor] = {}


def neighbor_matrix(grid_hw: tuple[int, int]) -> Tensor:
    """``[N, N]`` bool, true for 8-neighbours on the ``H × W`` grid (CPU, cached)."""

    if grid_hw not in _NEIGHBORS:
        height, width = grid_hw
        rows = torch.arange(height * width) // width
        cols = torch.arange(height * width) % width
        dr = (rows[:, None] - rows[None, :]).abs()
        dc = (cols[:, None] - cols[None, :]).abs()
        _NEIGHBORS[grid_hw] = (dr <= 1) & (dc <= 1) & ~((dr == 0) & (dc == 0))
    return _NEIGHBORS[grid_hw]


def focus_share(scores: Tensor, params: FocusContextParams) -> float:
    n = scores.numel()
    top = max(1, math.ceil(params.peak_top * n))
    uniform = top / n
    if uniform >= 1:
        return params.share_max
    peak = float(torch.topk(torch.softmax(scores.float(), 0), top).values.sum())
    t = min(1.0, max(0.0, (peak - uniform) / (1 - uniform)))
    return params.share_min + (params.share_max - params.share_min) * t


def grow_regions(scores: Tensor, k: int, grid_hw: tuple[int, int], beta: float) -> Tensor:
    """Greedy region growing: ``k`` indices, adjacent tokens favoured by ``beta``."""

    z = scores.detach().float().cpu()
    z = (z - z.mean()) / z.std().clamp_min(1e-6) if z.numel() > 1 else z
    neighbors = neighbor_matrix(grid_hw)
    selected = torch.zeros(z.numel(), dtype=torch.bool)
    adjacent = torch.zeros(z.numel(), dtype=torch.bool)
    for _ in range(min(k, z.numel())):
        candidate = z + beta * adjacent.float()
        candidate[selected] = -math.inf
        index = int(candidate.argmax())
        selected[index] = True
        adjacent |= neighbors[index]
    return torch.nonzero(selected).flatten()


def context_tokens(quality: Tensor, grid_hw: tuple[int, int], count: int, exclude: Tensor) -> Tensor:
    """Up to ``count`` tokens, at most one per coarse cell, skipping cells that hold excluded tokens."""

    n = quality.numel()
    if count <= 0:
        return torch.zeros(0, dtype=torch.long)
    height, width = grid_hw
    block = max(1, math.floor(math.sqrt(n / count)))
    index = torch.arange(n)
    cell = (index // width // block) * (-(-width // block)) + (index % width) // block
    q = quality.detach().float().cpu()
    taken = torch.zeros(n, dtype=torch.bool)
    taken[exclude.cpu()] = True
    covered = set(cell[taken].tolist())
    best: dict[int, int] = {}
    for i in torch.argsort(q, descending=True).tolist():
        c = int(cell[i])
        if not taken[i] and c not in covered and c not in best:
            best[c] = i
    chosen = sorted(best.values(), key=lambda i: -float(q[i]))[:count]
    if len(chosen) < count:
        free = [i for i in torch.argsort(q, descending=True).tolist() if not taken[i] and i not in set(chosen)]
        chosen += free[: count - len(chosen)]
    return torch.tensor(sorted(chosen), dtype=torch.long)


class FocusContextPruner(ScoredDSFoldPruner):
    """``head`` scores image-level quality (context); ``focus_head`` scores question-specific relevance."""

    name = "pp3_focus_context"

    def __init__(
        self,
        head: nn.Module,
        focus_head: nn.Module,
        *,
        params: DSFoldParams | None = None,
        compressor: FeatureCompressor | None = None,
        focus_params: FocusContextParams | None = None,
    ) -> None:
        super().__init__(head, params=params, compressor=compressor)
        self.focus_head = focus_head.eval()
        self.focus_params = focus_params or FocusContextParams()

    def with_focus_params(self, **overrides: Any) -> FocusContextPruner:
        return FocusContextPruner(
            self.head, self.focus_head, params=self.base.params, compressor=self.compressor,
            focus_params=self.focus_params.with_overrides(**overrides),
        )

    @torch.no_grad()
    def focus_scores(self, x: PrunerInput) -> Tensor:
        """Raw focus-head scores ``[N]`` (higher = more specific to this question)."""

        if x.ratio is None:
            raise ValueError("PP3 uses a fixed keep ratio.")
        scores = self.base.scores(x)
        features = build_token_features(
            x, scores, stopwords=self.base.params.stopwords, ratio=x.ratio, compressor=self.compressor
        )
        device = next(self.focus_head.parameters()).device
        return self.focus_head(features.to(device)).cpu()

    @torch.no_grad()
    def select(self, x: PrunerInput) -> PrunerOutput:
        scores, quality = self.token_quality(x)
        focus = self.focus_scores(x)
        k = budget_for_ratio(x.ratio, x.num_tokens)
        share = focus_share(focus, self.focus_params)
        k_focus = min(k, max(1, round(share * k)))
        focus_keep = grow_regions(focus, k_focus, x.grid_hw, self.focus_params.beta)
        count = k - focus_keep.numel()
        if self.focus_params.context_dpp >= 0.5 and count > 0:
            free = torch.ones(x.num_tokens, dtype=torch.bool)
            free[focus_keep] = False
            free_index = torch.nonzero(free).flatten().to(quality.device)
            order = select_order(
                scores.v_unit[free_index], quality[free_index], count,
                n_cap=self.base.params.n_cap, eps=self.base.params.d2_eps,
            )
            context_keep = free_index[order.keep_indices(count)].cpu()
        else:
            context_keep = context_tokens(quality, x.grid_hw, count, focus_keep)
        keep = torch.sort(torch.cat([focus_keep, context_keep])).values.to(x.v.device)
        output = self.base.output_for_keep(x, scores, keep, method=self.name, ratio=x.ratio)
        output.metadata.update({
            "selector": "focus_context",
            "focus_share": share,
            "focus_indices": focus_keep.tolist(),
        })
        return output


def build_pruner(config: dict[str, Any], *, project_root: Path, device: str) -> FocusContextPruner:
    params = load_dsfold_params(project_root / config["dsfold_params"])
    compressor = FeatureCompressor.load(project_root / config["compressor"]) if config.get("compressor") else None
    return FocusContextPruner(
        load_score_head(project_root / config["checkpoint"], device=device),
        load_score_head(project_root / config["focus_checkpoint"], device=device),
        params=params,
        compressor=compressor,
        focus_params=FocusContextParams().with_overrides(**(config.get("focus_context") or {})),
    )
