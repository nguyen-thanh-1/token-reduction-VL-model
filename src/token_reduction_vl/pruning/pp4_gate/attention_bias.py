"""Soft token removal through an additive attention bias (``method.md`` §4).

The bias ``log(z_i + ε)`` is added to the logits of every query attending to
visual key ``i`` (all layers, all heads). With ``z_i = 0`` and ``ε = 1e-4`` the
bias is ≈ −9.2, close to deleting the token, yet the mask stays differentiable.
Whether Transformers 5.x accepts this 4D float mask with SDPA must be checked
with ``scripts/pp4/spike_attention_bias.py`` before training.
"""

from __future__ import annotations

import torch
from torch import Tensor


def gated_attention_mask(
    seq_len: int,
    image_positions: Tensor,
    gates: Tensor,
    *,
    eps: float = 1e-4,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``[1, 1, L, L]`` additive mask: causal ``0 / min`` plus ``log(z + ε)`` on visual key columns."""

    device = gates.device
    allowed = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()
    column_bias = torch.zeros(seq_len, device=device, dtype=torch.float32)
    column_bias = column_bias.index_put((image_positions.to(device),), torch.log(gates.float() + eps))
    blocked = torch.finfo(dtype).min
    mask = torch.where(allowed, column_bias.unsqueeze(0).expand(seq_len, -1), torch.full((), blocked, device=device))
    return mask.to(dtype).view(1, 1, seq_len, seq_len)
