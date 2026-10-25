"""PP3 pruner: DSFold (set D) whose S1 quality comes from a learned score head."""

from __future__ import annotations

import torch
from torch import nn

from token_reduction_vl.pruning.base import PrunerInput, PrunerOutput, budget_for_ratio

from .dsfold_base import DSFoldBase, DSFoldParams
from .features import FeatureCompressor, build_token_features
from .models import quality_from_scores


class ScoredDSFoldPruner:
    name = "pp3_scorer"

    def __init__(
        self,
        head: nn.Module,
        *,
        params: DSFoldParams | None = None,
        compressor: FeatureCompressor | None = None,
    ) -> None:
        self.head = head.eval()
        self.base = DSFoldBase(params)
        self.compressor = compressor

    @torch.no_grad()
    def select(self, x: PrunerInput) -> PrunerOutput:
        if x.ratio is None:
            raise ValueError("PP3 uses a fixed keep ratio.")
        scores = self.base.scores(x)
        features = build_token_features(
            x, scores, stopwords=self.base.params.stopwords, ratio=x.ratio, compressor=self.compressor
        )
        device = next(self.head.parameters()).device
        quality = quality_from_scores(self.head(features.to(device))).to(scores.v_unit.device)
        k = budget_for_ratio(x.ratio, x.num_tokens)
        run = self.base.run(x, k, quality=quality, method=self.name)
        return run.output
