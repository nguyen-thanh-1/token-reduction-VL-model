"""Build the PP4 pruner from ``configs/methods/pp4_gate.yaml`` (used by ``scripts/common/run_pruned.py``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .dsfold_base import load_dsfold_params
from .pruner import GatedDSFoldPruner
from .train import load_gate_net


def build_pruner(config: dict[str, Any], *, project_root: Path, device: str) -> GatedDSFoldPruner:
    params = load_dsfold_params(project_root / config["dsfold_params"])
    net = load_gate_net(project_root / config["checkpoint"], device=device)
    return GatedDSFoldPruner(net, params=params, min_ratio=float(config.get("min_ratio", 1 / 18)))
