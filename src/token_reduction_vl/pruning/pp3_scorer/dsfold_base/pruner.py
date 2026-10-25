"""DSFold (PP1) assembled from S1–S4, with a hook to replace the quality score."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from token_reduction_vl.pruning.base import PrunerInput, PrunerOutput, budget_for_ratio

from .dpp import SelectionResult, select_order
from .fold import assign_dropped, dropped_indices, fold_deepstack
from .params import DSFoldParams
from .scoring import ScoreBundle, compute_scores


@dataclass(frozen=True)
class DSFoldRun:
    scores: ScoreBundle
    selection: SelectionResult
    output: PrunerOutput


class DSFoldBase:
    """PP1 with parameter set D. ``quality`` may be overridden (PP3) per call."""

    name = "dsfold_base"

    def __init__(self, params: DSFoldParams | None = None) -> None:
        self.params = params or DSFoldParams()

    def scores(self, x: PrunerInput) -> ScoreBundle:
        return compute_scores(x, self.params)

    def order(self, scores: ScoreBundle, k: int, quality: Tensor | None = None) -> SelectionResult:
        return select_order(
            scores.v_unit,
            scores.quality if quality is None else quality,
            k,
            n_cap=self.params.n_cap,
            eps=self.params.d2_eps,
        )

    def fold(self, x: PrunerInput, scores: ScoreBundle, keep: Tensor) -> list[Tensor]:
        drop = dropped_indices(x.num_tokens, keep)
        owner = assign_dropped(scores.v_unit, keep, drop, x.grid_hw, self.params.lam)
        return fold_deepstack(
            list(x.deepstack),
            scores.v_unit,
            keep,
            drop,
            owner,
            gamma=self.params.gamma,
            tau=self.params.tau,
        )

    def output_for_keep(
        self, x: PrunerInput, scores: ScoreBundle, keep: Tensor, *, method: str, ratio: float | None
    ) -> PrunerOutput:
        return PrunerOutput(
            keep_indices=keep,
            v_kept=x.v[keep],
            deepstack_kept=self.fold(x, scores, keep),
            metadata={
                "K": int(keep.numel()),
                "N": x.num_tokens,
                "method": method,
                "ratio_target": ratio,
                "dsfold_params": self.params.version,
            },
        )

    def run(self, x: PrunerInput, k: int, *, quality: Tensor | None = None, method: str | None = None) -> DSFoldRun:
        scores = self.scores(x)
        selection = self.order(scores, k, quality)
        keep = selection.keep_indices()
        output = self.output_for_keep(x, scores, keep, method=method or self.name, ratio=x.ratio)
        output.metadata["selector"] = selection.method
        return DSFoldRun(scores=scores, selection=selection, output=output)

    def select(self, x: PrunerInput) -> PrunerOutput:
        if x.ratio is None:
            raise ValueError("DSFoldBase needs a fixed keep ratio.")
        return self.run(x, budget_for_ratio(x.ratio, x.num_tokens)).output
