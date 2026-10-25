"""Route-B ranking loss and ranking metrics (``method.md`` §4.2, §6)."""

from __future__ import annotations

import torch
from torch import Tensor


def listnet_loss(scores: Tensor, saliency: Tensor, temperature: float = 0.5) -> Tensor:
    """``−Σ_i P_i log Q_i`` with ``P = softmax((sal / mean sal) / T)``, ``Q = softmax(s)``."""

    normalized = saliency.float() / saliency.float().mean().clamp_min(1e-12)
    target = torch.softmax(normalized / temperature, dim=0)
    return -(target * torch.log_softmax(scores.float(), dim=0)).sum()


def recall_at_k(scores: Tensor, saliency: Tensor, k: int) -> float:
    """Fraction of the top-``k`` saliency tokens that are also in the top-``k`` scores."""

    k = max(1, min(k, int(scores.numel())))
    top_scores = set(torch.topk(scores, k).indices.tolist())
    top_saliency = set(torch.topk(saliency, k).indices.tolist())
    return len(top_scores & top_saliency) / k


def spearman(a: Tensor, b: Tensor) -> float:
    if a.numel() < 2:
        return 0.0
    rank_a = torch.argsort(torch.argsort(a)).float()
    rank_b = torch.argsort(torch.argsort(b)).float()
    rank_a = rank_a - rank_a.mean()
    rank_b = rank_b - rank_b.mean()
    return float((rank_a * rank_b).sum() / (rank_a.norm() * rank_b.norm()).clamp_min(1e-12))
