"""DSFold parameter set D (PP1 ``method.md`` §8.0)."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from token_reduction_vl.config import load_yaml

DEFAULT_STOPWORDS = (
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "in", "on", "at", "to",
    "for", "with", "and", "or", "what", "which", "this", "that", "these", "those",
    "it", "its", "there", "do", "does", "?", ".", ",", ":", ";", "!",
)


@dataclass(frozen=True)
class DSFoldParams:
    version: str = "D-1.0"
    alpha: float = 1.0
    beta: float = 0.5
    ms_variant: str = "A"
    lam: float = 0.5
    gamma: float = 0.5
    tau: float = 0.1
    n_cap: int = 4096
    z_clip: float = 3.0
    d2_eps: float = 1e-6
    stopwords: tuple[str, ...] = DEFAULT_STOPWORDS

    def __post_init__(self) -> None:
        if self.ms_variant not in {"A", "B", "C"}:
            raise ValueError(f"ms_variant must be A, B or C, got {self.ms_variant!r}.")
        if self.tau <= 0:
            raise ValueError("tau must be positive.")

    def with_overrides(self, **overrides: Any) -> DSFoldParams:
        return replace(self, **overrides)


def load_dsfold_params(path: Path) -> DSFoldParams:
    """Load ``configs/dsfold_D-*.yaml``; unknown keys are rejected."""

    raw = load_yaml(path)
    allowed = {item.name for item in fields(DSFoldParams)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise KeyError(f"Unknown DSFold parameters in {path}: {unknown}")
    if "stopwords" in raw:
        raw["stopwords"] = tuple(str(word) for word in raw["stopwords"])
    return DSFoldParams(**raw)
