"""PP4 pruner: learned hard-concrete gates choose K and the tokens; DSFold folds the rest."""

from __future__ import annotations

import torch

from token_reduction_vl.pruning.base import PrunerInput, PrunerOutput

from .dsfold_base import DSFoldBase, DSFoldParams
from .features import build_gate_features
from .hard_concrete import keep_from_gates, open_probability
from .model import GateNet


class GatedDSFoldPruner:
    name = "pp4_gate"

    def __init__(self, net: GateNet, *, params: DSFoldParams | None = None, min_ratio: float = 1 / 18) -> None:
        self.net = net.eval()
        self.base = DSFoldBase(params)
        self.min_ratio = min_ratio

    @torch.no_grad()
    def select(self, x: PrunerInput) -> PrunerOutput:
        scores = self.base.scores(x)
        features = build_gate_features(x, scores, stopwords=self.base.params.stopwords)
        device = next(self.net.parameters()).device
        log_alpha = self.net(_to_device(features, device)).to(x.v.device)
        keep = keep_from_gates(log_alpha, self.min_ratio)
        output = self.base.output_for_keep(x, scores, keep, method=self.name, ratio=None)
        output.metadata["expected_ratio"] = float(open_probability(log_alpha).mean())
        return output


def _to_device(features, device: torch.device):
    from dataclasses import replace

    return replace(
        features,
        v=features.v.to(device),
        f=[level.to(device) for level in features.f],
        z_rel=features.z_rel.to(device),
        z_ms=features.z_ms.to(device),
        positions=features.positions.to(device),
        q=features.q.to(device),
    )
