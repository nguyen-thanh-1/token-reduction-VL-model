"""S1 — question relevance, multi-level saliency and quality (PP1 §3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from token_reduction_vl.pruning.base import PrunerInput

from .params import DSFoldParams


@dataclass(frozen=True)
class ScoreBundle:
    v_unit: Tensor  # [N, D] fp32, unit rows
    f_hat: list[Tensor]  # 3 × [N, D] fp32, per-level standardized DeepStack
    rel: Tensor  # [N]
    ms: Tensor  # [N]
    z_rel: Tensor  # [N]
    z_ms: Tensor  # [N]
    quality: Tensor  # [N], > 0


def question_embeddings(x: PrunerInput, stopwords: tuple[str, ...]) -> Tensor | None:
    """Embeddings of question tokens after stopword filtering (PP1 §3.1)."""

    ids = x.question_ids
    if ids.numel() == 0:
        return None
    stop = {word.lower() for word in stopwords}
    keep = [index for index, token in enumerate(x.question_tokens) if token.strip().lower() not in stop]
    if keep:
        ids = ids[torch.tensor(keep, dtype=torch.long, device=ids.device)]
    weight = x.embed_tokens.weight
    return weight[ids.to(weight.device)].detach().float()


def unit_rows(matrix: Tensor) -> Tensor:
    return matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def relevance(v_unit: Tensor, q: Tensor | None) -> Tensor:
    """``rel_i = max_j cos(v_i, q_j)``; zeros when there is no question text."""

    if q is None or q.numel() == 0:
        return torch.zeros(v_unit.shape[0], device=v_unit.device)
    return (v_unit @ unit_rows(q.to(v_unit.device)).T).max(dim=1).values


def standardize_levels(levels: list[Tensor]) -> list[Tensor]:
    """``f̂_l = (f_l − mean_tokens(f_l)) / std(f_l − mean)``; one population std per level."""

    out = []
    for level in levels:
        centered = level.float() - level.float().mean(dim=0, keepdim=True)
        out.append(centered / (centered.std(unbiased=False) + 1e-6))
    return out


def multilevel_saliency(f_hat: list[Tensor], variant: str) -> Tensor:
    if variant == "A":
        return (f_hat[2] - f_hat[1]).norm(dim=-1) + (f_hat[1] - f_hat[0]).norm(dim=-1)
    norms = torch.stack([level.norm(dim=-1) for level in f_hat])
    if variant == "B":
        return norms.std(dim=0, unbiased=False)
    if variant == "C":
        return norms.mean(dim=0)
    raise ValueError(f"Unknown ms variant {variant!r}.")


def zscore(values: Tensor, clip: float) -> Tensor:
    if values.numel() < 2:
        return torch.zeros_like(values)
    std = values.std(unbiased=False)
    return ((values - values.mean()) / (std + 1e-6)).clamp(-clip, clip)


def compute_scores(x: PrunerInput, params: DSFoldParams) -> ScoreBundle:
    v_unit = unit_rows(x.v.float())
    f_hat = standardize_levels(list(x.deepstack))
    rel = relevance(v_unit, question_embeddings(x, params.stopwords))
    ms = multilevel_saliency(f_hat, params.ms_variant)
    z_rel = zscore(rel, params.z_clip)
    z_ms = zscore(ms, params.z_clip)
    quality = torch.exp(params.alpha * z_rel + params.beta * z_ms)
    return ScoreBundle(v_unit=v_unit, f_hat=f_hat, rel=rel, ms=ms, z_rel=z_rel, z_ms=z_ms, quality=quality)
