"""PP4 — DSFold with learned hard-concrete gates (``pp4-dsfold-gate/``)."""

from .attention_bias import gated_attention_mask
from .features import GateFeatures, build_gate_features
from .hard_concrete import deterministic_gates, keep_from_gates, open_probability, sample_gates
from .losses import GateSchedule, gate_objective
from .model import GateNet
from .pruner import GatedDSFoldPruner

__all__ = [
    "GateFeatures",
    "GateNet",
    "GateSchedule",
    "GatedDSFoldPruner",
    "build_gate_features",
    "deterministic_gates",
    "gate_objective",
    "gated_attention_mask",
    "keep_from_gates",
    "open_probability",
    "sample_gates",
]
