"""External selection baselines in the shared pruner interface (ADR 0005).

Both reuse DSFold's output path so that only the selection differs from PP3;
pass parameters with ``gamma=0`` (the default here) to drop tokens without
folding them, as the published methods do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from token_reduction_vl.pruning.base import PrunerInput, PrunerOutput, budget_for_ratio

from .dsfold_base import DSFoldBase, DSFoldParams

if TYPE_CHECKING:
    from token_reduction_vl.models.qwen3_vl_pruned import EncodedSample, PrunedQwen3VL

NO_FOLD = DSFoldParams().with_overrides(gamma=0.0)
FASTV_LAYER = 1  # 0-based: FastV ranks with the attention of layer 2


def divprune_order(v: Tensor, k: int) -> Tensor:
    """Greedy max–min diversity on cosine distance; returns ``k`` indices in pick order."""

    unit = torch.nn.functional.normalize(v.float(), dim=-1)
    distance = 1.0 - unit @ unit.T
    k = min(k, int(v.shape[0]))
    first = int(distance.mean(1).argmax())
    order = [first]
    selected = torch.zeros(v.shape[0], dtype=torch.bool, device=v.device)
    selected[first] = True
    min_dist = distance[first].clone()
    for _ in range(1, k):
        j = int(min_dist.masked_fill(selected, float("-inf")).argmax())
        order.append(j)
        selected[j] = True
        min_dist = torch.minimum(min_dist, distance[j])
    return torch.tensor(order, dtype=torch.long, device=v.device)


class DivPrunePruner:
    """DivPrune (Alvar et al., 2025): the ``K`` most mutually diverse visual tokens."""

    name = "divprune"

    def __init__(self, params: DSFoldParams | None = None) -> None:
        self.base = DSFoldBase(params or NO_FOLD)

    @torch.no_grad()
    def select(self, x: PrunerInput) -> PrunerOutput:
        if x.ratio is None:
            raise ValueError("DivPrunePruner needs a fixed keep ratio.")
        k = budget_for_ratio(x.ratio, x.num_tokens)
        keep = torch.sort(divprune_order(x.v, k)).values
        return self.base.output_for_keep(x, self.base.scores(x), keep, method=self.name, ratio=x.ratio)


def cdpruner_relevance(x: PrunerInput) -> Tensor:
    """``r̃ ∈ [0, 1]^N``: min-max normalised ``cos(v_i, mean of the instruction's token embeddings)``.

    CDPruner (Zhang et al., 2025, Eq. 5–6) for models without a paired text
    encoder: projector output against the LLM embeddings of the instruction.
    """

    v = torch.nn.functional.normalize(x.v.float(), dim=-1)
    if x.question_ids.numel() == 0:
        return torch.ones(v.shape[0], device=v.device)
    weight = x.embed_tokens.weight
    q = weight[x.question_ids.to(weight.device)].detach().float().mean(dim=0).to(v.device)
    r = v @ torch.nn.functional.normalize(q, dim=0)
    span = r.max() - r.min()
    return (r - r.min()) / span if float(span) > 0 else torch.ones_like(r)


class CDPrunerPruner:
    """CDPruner (Zhang et al., 2025): DPP on ``diag(r̃)·S·diag(r̃)`` with instruction relevance ``r̃``.

    Training-free conditional DPP, no fold. It shares PP3's greedy DPP (and its
    weighted max–min fallback above ``n_cap`` tokens), so the comparison with PP3
    isolates the quality term: hand-crafted relevance here, the learned score head there.
    """

    name = "cdpruner"

    def __init__(self, params: DSFoldParams | None = None) -> None:
        self.base = DSFoldBase(params or NO_FOLD)

    @torch.no_grad()
    def select(self, x: PrunerInput) -> PrunerOutput:
        if x.ratio is None:
            raise ValueError("CDPrunerPruner needs a fixed keep ratio.")
        k = budget_for_ratio(x.ratio, x.num_tokens)
        scores = self.base.scores(x)
        keep = self.base.order(scores, k, cdpruner_relevance(x)).keep_indices()
        return self.base.output_for_keep(x, scores, keep, method=self.name, ratio=x.ratio)


def _attention_rows(module, hidden: Tensor, cos: Tensor, sin: Tensor, start: int, cols: Tensor, chunk: int) -> Tensor:
    """Sum over rows ``start:`` of the head-averaged causal attention on columns ``cols``.

    Recomputes the layer's queries and keys and walks the rows in blocks of
    ``chunk``, so the ``[H, L, L]`` matrix that eager attention would build is
    never held at once (MME images reach ~14k tokens). Matmul in the model
    dtype and softmax in float32, as ``eager_attention_forward`` does."""

    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

    shape = (*hidden.shape[:-1], -1, module.head_dim)
    q = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
    k = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k = repeat_kv(k, module.num_key_value_groups)[0]  # [H, L, D]
    q = q[0]
    length = int(k.shape[1])
    total = torch.zeros(cols.numel(), dtype=torch.float32, device=k.device)
    for lo in range(start, length, chunk):
        hi = min(lo + chunk, length)
        logits = torch.matmul(q[:, lo:hi], k.transpose(1, 2)) * module.scaling  # [H, R, L]
        future = torch.arange(length, device=k.device)[None, :] > torch.arange(lo, hi, device=k.device)[:, None]
        logits = logits.masked_fill(future, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
        total += weights[:, :, cols].float().mean(0).sum(0)
        del logits, weights
    return total


@torch.no_grad()
def received_attention(
    runner: PrunedQwen3VL, encoded: EncodedSample, layer: int = FASTV_LAYER, chunk: int = 512
) -> Tensor:
    """``[N]`` mean attention each visual token receives from all later positions in ``layer``
    of the full-token prefill (heads averaged). The forward stops before that layer's
    attention runs; the weights are rebuilt in row blocks of ``chunk`` (see ``_attention_rows``)."""

    n = encoded.num_visual
    selection = PrunerOutput(
        keep_indices=torch.arange(n, device=encoded.v.device),
        v_kept=encoded.v,
        deepstack_kept=list(encoded.deepstack),
        metadata={"K": n, "N": n, "method": "none", "ratio_target": 1.0},
    )
    prepared = runner.prepare(encoded, selection)
    cols = encoded.image_positions.to(runner.device)
    start = int(cols.min())
    captured: list[Tensor] = []

    class _Stop(Exception):
        pass

    def hook(module, _args, kwargs) -> None:
        cos, sin = kwargs["position_embeddings"]
        received = _attention_rows(module, kwargs["hidden_states"], cos, sin, start, cols, chunk)
        # causal: row r sees column c only when r >= c
        length = int(kwargs["hidden_states"].shape[1])
        later = (torch.arange(start, length, device=cols.device)[:, None] >= cols[None, :]).sum(0)
        captured.append(received / later.clamp_min(1))
        raise _Stop

    lm = runner.language_model
    handle = lm.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        lm(
            inputs_embeds=prepared.inputs_embeds,
            position_ids=prepared.position_ids,
            attention_mask=torch.ones(prepared.inputs_embeds.shape[:2], dtype=torch.long, device=runner.device),
            visual_pos_masks=prepared.visual_pos_masks,
            deepstack_visual_embeds=prepared.deepstack,
            use_cache=False,
        )
    except _Stop:
        pass
    finally:
        handle.remove()
    return captured[0]


class FastVPruner:
    """FastV-style selection (Chen et al., 2024) at the LLM input: top-``K`` visual tokens by
    the attention they receive in LLM layer 2. Call ``prepare_sample`` before ``select``."""

    name = "fastv"

    def __init__(self, params: DSFoldParams | None = None, layer: int = FASTV_LAYER) -> None:
        self.base = DSFoldBase(params or NO_FOLD)
        self.layer = layer
        self.attention: Tensor | None = None

    def prepare_sample(self, runner: PrunedQwen3VL, encoded: EncodedSample) -> None:
        self.attention = received_attention(runner, encoded, self.layer)

    @torch.no_grad()
    def select(self, x: PrunerInput) -> PrunerOutput:
        if x.ratio is None:
            raise ValueError("FastVPruner needs a fixed keep ratio.")
        if self.attention is None or self.attention.numel() != x.num_tokens:
            raise ValueError("FastVPruner.prepare_sample must run on this sample first.")
        k = budget_for_ratio(x.ratio, x.num_tokens)
        keep = torch.sort(torch.topk(self.attention.to(x.v.device), k).indices).values
        return self.base.output_for_keep(x, self.base.scores(x), keep, method=self.name, ratio=x.ratio)
