"""Route-B labels: gradient × input saliency of every visual token (``method.md`` §4.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from token_reduction_vl.pruning.base import PrunerOutput

if TYPE_CHECKING:
    from token_reduction_vl.models.qwen3_vl_pruned import EncodedSample, PrunedQwen3VL

MAX_ANSWER_TOKENS = 8


def full_token_answer(runner: PrunedQwen3VL, encoded: EncodedSample) -> Tensor:
    """Greedy full-token answer ids, without EOS, truncated to ``MAX_ANSWER_TOKENS``."""

    n = encoded.num_visual
    selection = PrunerOutput(
        keep_indices=torch.arange(n, device=encoded.v.device),
        v_kept=encoded.v,
        deepstack_kept=list(encoded.deepstack),
        metadata={"K": n, "N": n, "method": "none", "ratio_target": 1.0},
    )
    ids, _, _ = runner.greedy_decode(runner.prepare(encoded, selection))
    ids = [token for token in ids if token not in runner.eos_token_ids][:MAX_ANSWER_TOKENS]
    return torch.tensor(ids, dtype=torch.long, device=encoded.v.device)


def gradient_x_input_saliency(runner: PrunedQwen3VL, encoded: EncodedSample, answer_ids: Tensor) -> Tensor:
    """``sal_i = |⟨∂log p/∂v_i, v_i⟩| + Σ_l |⟨∂log p/∂f_l[i], f_l[i]⟩|`` with ``log p`` teacher-forced.

    Only input gradients are computed; model weights stay frozen.
    """

    if answer_ids.numel() == 0:
        raise ValueError("Saliency needs at least one answer token.")
    n = encoded.num_visual
    v = encoded.v.detach().clone().requires_grad_(True)
    levels = [level.detach().clone().requires_grad_(True) for level in encoded.deepstack]
    selection = PrunerOutput(
        keep_indices=torch.arange(n, device=v.device),
        v_kept=v,
        deepstack_kept=levels,
        metadata={"K": n, "N": n, "method": "none", "ratio_target": 1.0},
    )
    with torch.enable_grad():
        prepared = runner.prepare(encoded, selection)
        logits = runner.teacher_forced_logits(prepared, answer_ids)
        log_prob = torch.log_softmax(logits, dim=-1).gather(1, answer_ids.view(-1, 1).to(logits.device)).sum()
        grads = torch.autograd.grad(log_prob, [v, *levels])
    saliency = (grads[0].float() * v.float()).sum(dim=-1).abs()
    for grad, level in zip(grads[1:], levels, strict=True):
        saliency = saliency + (grad.float() * level.float()).sum(dim=-1).abs()
    return saliency.detach()
