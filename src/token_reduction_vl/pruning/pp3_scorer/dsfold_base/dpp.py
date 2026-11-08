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
    """Greedy MAP for ``L = diag(q)·S·diag(q)``, ``S = v_unit v_unitᵀ`` (Chen et al., 2018).

    On CPU (and for small ``K``) this is :func:`greedy_dpp_reference`. On CUDA the
    loop is launch-bound (a dozen tiny kernels per pick), so it keeps every index
    on the device, runs all ``K`` steps in place on fixed-shape buffers and
    replays one captured graph of ``GRAPH_UNROLL`` steps. The stop rule
    (``d2 < eps`` → fill the rest by quality) is applied once at the end.
    Cholesky rows beyond the current step are zero, so the fixed-shape product
    has the same terms as the reference's growing slice; only their summation
    order differs, which can flip a near-tie late in a long run (seen once, at
    N = 3072 and K = 1536, in the conformance fixtures).
    """

    n = int(v_unit.shape[0])
    k = min(k, n)
    device = v_unit.device
    if device.type != "cuda" or k < GRAPH_MIN_STEPS:
        return greedy_dpp_reference(v_unit, quality, k, eps)
    quality = quality.float()
    state = _DppState(v_unit.float() @ v_unit.float().T, quality, k, eps)
    _run_graphed(state, k)
    order, gains = state.order[:k], state.gains[:k]
    stops = torch.nonzero(~(gains >= eps), as_tuple=False)  # also catches NaN
    if stops.numel():
        stop = int(stops[0])
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[order[:stop]] = True
        order = torch.cat([order[:stop], _fill_by_quality(quality, mask, k - stop)])
        gains = torch.cat([gains[:stop], torch.zeros(k - stop, device=device)])
    return SelectionResult(order=order, gains=gains, method="dpp")


GRAPH_UNROLL = 16
"""Greedy steps per captured CUDA graph; the buffers are padded so extra steps are harmless."""
GRAPH_MIN_STEPS = 48
"""Below this K, capturing the graph costs more than launching the kernels one by one."""


class _DppState:
    """Fixed-shape, in-place buffers of the greedy DPP; ``step`` has no host sync."""

    def __init__(self, similarity: Tensor, quality: Tensor, k: int, eps: float) -> None:
        n, device = int(quality.shape[0]), quality.device
        rows = k + GRAPH_UNROLL
        self.similarity, self.quality = similarity, quality
        self.d2 = quality.square().clone()
        self.cholesky = torch.zeros((rows, n), device=device)  # row = step
        self.selected = torch.zeros(n, dtype=torch.bool, device=device)
        self.order = torch.zeros(rows, dtype=torch.long, device=device)
        self.gains = torch.zeros(rows, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)
        self.floor = torch.tensor(eps, device=device)

    def reset(self) -> None:
        self.d2.copy_(self.quality.square())
        self.cholesky.zero_()
        self.selected.zero_()
        self.position.zero_()

    def step(self) -> None:
        j = self.d2.masked_fill(self.selected, float("-inf")).argmax().view(1)
        gain = self.d2.index_select(0, j)
        self.order.index_copy_(0, self.position, j)
        self.gains.index_copy_(0, self.position, gain)
        self.selected.index_fill_(0, j, True)
        row = self.quality.index_select(0, j) * self.similarity.index_select(0, j)[0] * self.quality
        e = (row - self.cholesky.index_select(1, j)[:, 0] @ self.cholesky) / torch.sqrt(torch.maximum(gain, self.floor))
        self.cholesky.index_copy_(0, self.position, e.view(1, -1))
        self.d2.sub_(e.square())
        self.position.add_(1)


def _run_graphed(state: _DppState | _MaxMinState, k: int) -> None:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # warm-up outside the graph, as CUDA graphs require
        for _ in range(GRAPH_UNROLL):
            state.step()
    torch.cuda.current_stream().wait_stream(stream)
    state.reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH_UNROLL):
            state.step()
    for _ in range(-(-k // GRAPH_UNROLL)):
        graph.replay()


def greedy_dpp_reference(v_unit: Tensor, quality: Tensor, k: int, eps: float = 1e-6) -> SelectionResult:
    """Step-by-step greedy MAP with a host sync per pick; kept as the test oracle."""

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
    """``j = argmax quality_i · min_{s∈sel}(1 − cos(v_i, v_s))``; O(N) memory.

    Same picks as :func:`weighted_maxmin_reference`, run like :func:`greedy_dpp`
    (device-side indices, CUDA graph replay). ``min_dist`` starts at 2, the largest
    cosine distance, so the first pick is ``argmax quality`` (×2 is exact).
    """

    n = int(v_unit.shape[0])
    k = min(k, n)
    if v_unit.device.type != "cuda" or k < GRAPH_MIN_STEPS:
        return weighted_maxmin_reference(v_unit, quality, k)
    state = _MaxMinState(v_unit.float(), quality.float(), k)
    _run_graphed(state, k)
    return SelectionResult(order=state.order[:k], gains=None, method="maxmin")


class _MaxMinState:
    def __init__(self, v_unit: Tensor, quality: Tensor, k: int) -> None:
        n, device = int(quality.shape[0]), quality.device
        self.v_unit, self.quality = v_unit, quality
        self.min_dist = torch.full((n,), 2.0, device=device)
        self.selected = torch.zeros(n, dtype=torch.bool, device=device)
        self.order = torch.zeros(k + GRAPH_UNROLL, dtype=torch.long, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)

    def reset(self) -> None:
        self.min_dist.fill_(2.0)
        self.selected.zero_()
        self.position.zero_()

    def step(self) -> None:
        j = (self.quality * self.min_dist).masked_fill(self.selected, float("-inf")).argmax().view(1)
        self.order.index_copy_(0, self.position, j)
        self.selected.index_fill_(0, j, True)
        torch.minimum(self.min_dist, 1.0 - self.v_unit @ self.v_unit.index_select(0, j)[0], out=self.min_dist)
        self.position.add_(1)


def weighted_maxmin_reference(v_unit: Tensor, quality: Tensor, k: int) -> SelectionResult:
    """Step-by-step weighted max–min with a host sync per pick; kept as the test oracle."""

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
