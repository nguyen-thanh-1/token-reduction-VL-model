"""S3 Assign and S4 Fold into DeepStack (PP1 §5–6)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def grid_coordinates(num_tokens: int, grid_hw: tuple[int, int], device: torch.device) -> Tensor:
    """``[N, 2]`` float ``(row, col)`` of each token in raster order."""

    _, width = grid_hw
    index = torch.arange(num_tokens, device=device)
    return torch.stack([torch.div(index, width, rounding_mode="floor"), index % width], dim=1).float()


def dropped_indices(num_tokens: int, keep: Tensor) -> Tensor:
    mask = torch.ones(num_tokens, dtype=torch.bool, device=keep.device)
    mask[keep] = False
    return torch.nonzero(mask, as_tuple=False).flatten()


def assign_dropped(
    v_unit: Tensor, keep: Tensor, drop: Tensor, grid_hw: tuple[int, int], lam: float
) -> Tensor:
    """For each dropped token, the position (0..K−1) in ``keep`` of its owner.

    ``a(d) = argmax_k [cos(v_d, v_k) − λ·dist(d, k)]``, distance normalized by the grid diagonal.
    """

    if drop.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=keep.device)
    coords = grid_coordinates(int(v_unit.shape[0]), grid_hw, v_unit.device)
    diagonal = math.sqrt(grid_hw[0] ** 2 + grid_hw[1] ** 2)
    similarity = v_unit[drop] @ v_unit[keep].T
    distance = torch.cdist(coords[drop], coords[keep]) / max(diagonal, 1e-6)
    return (similarity - lam * distance).argmax(dim=1)


def fold_deepstack(
    levels: list[Tensor],
    v_unit: Tensor,
    keep: Tensor,
    drop: Tensor,
    owner: Tensor,
    *,
    gamma: float,
    tau: float,
) -> list[Tensor]:
    """``f'_l[k] = norm-match(f_l[k] + γ·Σ_{d∈A(k)} w_dk·f_l[d])``; returns 3 × ``[K, D]``.

    ``w_dk`` is a softmax over the members of ``A(k)`` of ``cos(v_d, v_k)/τ``.
    Tokens with an empty group keep ``f_l[k]`` unchanged.
    """

    if gamma == 0.0 or drop.numel() == 0:
        return [level[keep] for level in levels]
    k = int(keep.numel())
    logits = (v_unit[drop] * v_unit[keep][owner]).sum(dim=-1) / tau
    group_max = torch.full((k,), float("-inf"), device=logits.device).scatter_reduce(
        0, owner, logits, reduce="amax", include_self=True
    )
    weights = torch.exp(logits - group_max[owner])
    denominator = torch.zeros(k, device=logits.device).index_add(0, owner, weights)
    weights = weights / denominator[owner]
    has_group = (denominator > 0).unsqueeze(-1)

    folded = []
    for level in levels:
        base = level[keep].float()
        aggregate = torch.zeros_like(base).index_add(0, owner, weights.unsqueeze(-1) * level[drop].float())
        tilde = base + gamma * aggregate
        scale = base.norm(dim=-1, keepdim=True) / tilde.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        folded.append(torch.where(has_group, tilde * scale, base).to(level.dtype))
    return folded
