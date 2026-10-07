"""Method-agnostic pruning interface.

The baseline deliberately does not choose a pruning strategy. Future methods
can implement this interface without changing dataset or evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class PruningOutput:
    selected_tokens: Any
    selected_indices: Any = None
    retained_ratio: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class VisualTokenPruner(Protocol):
    name: str

    def select(
        self,
        visual_tokens: Any,
        *,
        question_embedding: Any = None,
        spatial_shape: tuple[int, int] | None = None,
    ) -> PruningOutput: ...


class NoOpPruner:
    """Reference implementation representing the full-token baseline."""

    name = "none"

    def select(
        self,
        visual_tokens: Any,
        *,
        question_embedding: Any = None,
        spatial_shape: tuple[int, int] | None = None,
    ) -> PruningOutput:
        token_count = len(visual_tokens)
        indices = list(range(token_count))
        return PruningOutput(
            selected_tokens=visual_tokens,
            selected_indices=indices,
            retained_ratio=1.0,
            metadata={"token_count": token_count},
        )
