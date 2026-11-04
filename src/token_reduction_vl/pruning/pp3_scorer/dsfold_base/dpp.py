"""S2 — greedy MAP DPP and weighted max–min selection (PP1 §4).

Both selectors return tokens in *selection order*, so the first ``K`` entries
of a run to ``K_max`` equal a run to ``K`` (nested sets, used by PP2/PP4).
Ties are broken by the smallest index so that every implementation agrees.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SelectionResult:
    order: Tensor  # [K] int64 in selection order
    gains: Tensor | None  # [K] d2 at each pick (DPP only); None for max–min
    method: str

    def keep_indices(self, k: int | None = None) -> Tensor:
        """Sorted indices of the first ``k`` picks (all picks when ``k`` is None)."""

        order = self.order if k is None else self.order[:k]
        return torch.sort(order).values


def _fill_by_quality(quality: Tensor, selected: Tensor, count: int) -> Tensor:
    remaining = torch.nonzero(~selected, as_tuple=False).flatten()
    ranked = torch.sort(-quality[remaining], stable=True).indices
    return remaining[ranked[:count]]


def greedy_dpp(v_unit: Tensor, quality: Tensor, k: int, eps: float = 1e-6) -> SelectionResult:
    """Greedy MAP for ``L = diag(q)·S·diag(q)``, ``S = v_unit v_unitᵀ`` (Chen et al., 2018)."""

    n = int(v_unit.shape[0])
    k = min(k, n)
    quality = quality.float()
    similarity = v_unit.float() @ v_unit.float().T
    d2 = quality.square().clone()
    cholesky = torch.zeros((n, k), device=v_unit.device)
    selected = torch.zeros(n, dtype=torch.bool, device=v_unit.device)
    order: list[int] = []
    gains: list[float] = []
    for step in range(k):
        j = int(d2.masked_fill(selected, float("-inf")).argmax())
        if float(d2[j]) < eps:
            filled = _fill_by_quality(quality, selected, k - step)
            order.extend(int(index) for index in filled)
            gains.extend([0.0] * int(filled.numel()))
            break
        order.append(j)
        gains.append(float(d2[j]))
        selected[j] = True
        row = quality[j] * similarity[j] * quality
        e = (row - cholesky[j, :step] @ cholesky[:, :step].T) / torch.sqrt(d2[j])
        cholesky[:, step] = e
        d2 = d2 - e.square()
    return SelectionResult(
        order=torch.tensor(order, dtype=torch.long, device=v_unit.device),
        gains=torch.tensor(gains, device=v_unit.device),
        method="dpp",
    )


def weighted_maxmin(v_unit: Tensor, quality: Tensor, k: int) -> SelectionResult:
    """``j = argmax quality_i · min_{s∈sel}(1 − cos(v_i, v_s))``; O(N) memory."""

    n = int(v_unit.shape[0])
    k = min(k, n)
    v_unit = v_unit.float()
    quality = quality.float()
    selected = torch.zeros(n, dtype=torch.bool, device=v_unit.device)
    first = int(quality.argmax())
    order = [first]
    selected[first] = True
    min_dist = 1.0 - v_unit @ v_unit[first]
    for _ in range(1, k):
        j = int((quality * min_dist).masked_fill(selected, float("-inf")).argmax())
        order.append(j)
        selected[j] = True
        min_dist = torch.minimum(min_dist, 1.0 - v_unit @ v_unit[j])
    return SelectionResult(
        order=torch.tensor(order, dtype=torch.long, device=v_unit.device), gains=None, method="maxmin"
    )


def select_order(v_unit: Tensor, quality: Tensor, k: int, *, n_cap: int, eps: float) -> SelectionResult:
    """Exact DPP up to ``n_cap`` visual tokens, weighted max–min above (PP1 §4.2b)."""

    if v_unit.shape[0] <= n_cap:
        return greedy_dpp(v_unit, quality, k, eps)
    return weighted_maxmin(v_unit, quality, k)
