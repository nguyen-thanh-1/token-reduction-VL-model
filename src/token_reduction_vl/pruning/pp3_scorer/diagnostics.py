"""Stage-1 diagnostic pruners for PP3 (ADR 0002): random selection and top-``q̂`` drop.

Both reuse DSFold's S3/S4 fold so that only the *selection* differs from PP3.
Pass parameters with ``gamma=0`` to remove dropped tokens without folding them.
"""

from __future__ import annotations

import torch

from token_reduction_vl.pruning.base import PrunerInput, PrunerOutput, budget_for_ratio

from .dsfold_base import DSFoldBase, DSFoldParams
from .pruner import ScoredDSFoldPruner


class RandomKeepPruner:
    """Keep ``K = budget(ratio)`` tokens drawn uniformly; reproducible for a fixed sample order."""

    name = "random"

    def __init__(self, params: DSFoldParams | None = None, seed: int = 0) -> None:
        self.base = DSFoldBase(params)
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def select(self, x: PrunerInput) -> PrunerOutput:
        if x.ratio is None:
            raise ValueError("RandomKeepPruner needs a fixed keep ratio.")
        k = budget_for_ratio(x.ratio, x.num_tokens)
        keep = torch.sort(torch.randperm(x.num_tokens, generator=self.generator)[:k]).values.to(x.v.device)
        return self.base.output_for_keep(x, self.base.scores(x), keep, method=self.name, ratio=x.ratio)


class TopScorePruner(ScoredDSFoldPruner):
    """Keep the ``K`` highest-``q̂`` tokens without DPP; ratio 0.9 drops the 10% lowest-scored."""

    name = "pp3_topk"

    @torch.no_grad()
    def select(self, x: PrunerInput) -> PrunerOutput:
        scores, quality = self.token_quality(x)
        k = budget_for_ratio(x.ratio, x.num_tokens)
        keep = torch.sort(torch.topk(quality, k).indices).values
        return self.base.output_for_keep(x, scores, keep, method=self.name, ratio=x.ratio)
