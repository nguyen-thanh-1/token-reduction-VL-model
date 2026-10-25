"""Hard-concrete gates (Louizos et al., ICLR 2018) — ``pp4-dsfold-gate/method.md`` §3."""

from __future__ import annotations

import math

import torch
from torch import Tensor

BETA = 2 / 3
GAMMA = -0.1
ZETA = 1.1


def initial_log_alpha(beta: float = BETA) -> float:
    """``log α`` for which ``P(z > 0) = 0.5`` (≈ −1.6 with the default constants)."""

    return beta * math.log(-GAMMA / ZETA)


def sample_gates(log_alpha: Tensor, beta: float = BETA, generator: torch.Generator | None = None) -> Tensor:
    """Reparameterized sample ``z ∈ [0, 1]`` with point masses at 0 and 1 (training)."""

    u = torch.rand(log_alpha.shape, generator=generator, device=log_alpha.device).clamp(1e-6, 1 - 1e-6)
    s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + log_alpha) / beta)
    return (s * (ZETA - GAMMA) + GAMMA).clamp(0.0, 1.0)


def open_probability(log_alpha: Tensor, beta: float = BETA) -> Tensor:
    """``π_i = P(z_i > 0) = σ(log α_i − β·log(−γ/ζ))``; ``E[K] = Σ π_i``."""

    return torch.sigmoid(log_alpha - beta * math.log(-GAMMA / ZETA))


def deterministic_gates(log_alpha: Tensor) -> Tensor:
    """Test-time gate ``ẑ = clip(σ(log α)·(ζ − γ) + γ, 0, 1)``."""

    return (torch.sigmoid(log_alpha) * (ZETA - GAMMA) + GAMMA).clamp(0.0, 1.0)


def keep_from_gates(log_alpha: Tensor, min_ratio: float = 1 / 18) -> Tensor:
    """Sorted indices with ``ẑ ≥ 0.5``; at least ``K_min = ⌈min_ratio·N⌉`` tokens."""

    n = int(log_alpha.numel())
    k_min = max(1, min(n, math.ceil(min_ratio * n - 1e-6)))
    keep = torch.nonzero(deterministic_gates(log_alpha) >= 0.5, as_tuple=False).flatten()
    if keep.numel() < k_min:
        keep = torch.topk(log_alpha, k_min).indices
    return torch.sort(keep).values
