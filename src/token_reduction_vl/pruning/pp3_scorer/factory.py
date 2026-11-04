"""Build the PP3 pruner from ``configs/methods/pp3_scorer*.yaml`` (used by ``scripts/common/run_pruned.py``).

With ``checkpoint: null`` the factory returns PP3's own copy of DSFold with
parameter set D, i.e. the reference that PP3 must beat at the same ratio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .dsfold_base import DSFoldBase, load_dsfold_params
from .features import FeatureCompressor
from .pruner import ScoredDSFoldPruner
from .train import load_score_head


def build_pruner(config: dict[str, Any], *, project_root: Path, device: str) -> ScoredDSFoldPruner | DSFoldBase:
    params = load_dsfold_params(project_root / config["dsfold_params"])
    if not config.get("checkpoint"):
        return DSFoldBase(params)
    compressor_path = config.get("compressor")
    compressor = FeatureCompressor.load(project_root / compressor_path) if compressor_path else None
    head = load_score_head(project_root / config["checkpoint"], device=device)
    return ScoredDSFoldPruner(head, params=params, compressor=compressor)
