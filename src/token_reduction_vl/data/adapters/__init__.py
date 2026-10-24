"""Dataset-specific adapters for the canonical VQA schema."""

from .gqa import iter_gqa_samples
from .mmb import iter_mmb_samples
from .mme import iter_mme_samples

__all__ = ["iter_gqa_samples", "iter_mmb_samples", "iter_mme_samples"]
