"""LLM text-to-image attention as a distillation target for the PP3 score head (ADR 0003).

The full-token model is run once, teacher-forced on an answer. Only this
forward switches the LLM to eager attention (the ViT and decoding keep their
faster kernels). Forward hooks on the chosen decoder layers read each head's
attention from selected text rows to the visual columns, renormalise it over
visual tokens and average it over heads, rows and layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from token_reduction_vl.pruning.base import PrunerOutput

if TYPE_CHECKING:
    from token_reduction_vl.models.qwen3_vl_pruned import EncodedSample, PrunedQwen3VL

DEFAULT_LAYERS = tuple(range(8, 20))


@dataclass(frozen=True)
class AttentionTeacher:
    answer: Tensor
    """``[N]`` attention from the last prompt position and the answer positions."""
    question: Tensor
    """``[N]`` attention from the question-text positions (equals ``answer`` if none were found)."""
    question_rows: int


def question_positions(input_ids: Tensor, image_positions: Tensor, question_ids: Tensor) -> Tensor:
    """Sequence positions of the question text, the longest prefix of ``question_ids``
    that follows ``<|vision_end|>``. Tokens may merge at the end (``?`` + ``\\n\\n``),
    so a mismatch ends the span rather than failing."""

    ids = input_ids.reshape(-1).tolist()
    question = question_ids.reshape(-1).tolist()
    start = int(image_positions.max()) + 2
    count = 0
    while count < len(question) and start + count < len(ids) and ids[start + count] == question[count]:
        count += 1
    return torch.arange(start, start + count, dtype=torch.long)


def attention_teacher(
    runner: PrunedQwen3VL,
    encoded: EncodedSample,
    answer_ids: Tensor,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
) -> AttentionTeacher:
    if answer_ids.numel() == 0:
        raise ValueError("The attention teacher needs at least one answer token.")
    decoder_layers = runner.language_model.layers
    if not layers or max(layers) >= len(decoder_layers) or min(layers) < 0:
        raise ValueError(f"Layers {layers} are outside the {len(decoder_layers)} decoder layers.")

    n = encoded.num_visual
    selection = PrunerOutput(
        keep_indices=torch.arange(n, device=encoded.v.device),
        v_kept=encoded.v,
        deepstack_kept=list(encoded.deepstack),
        metadata={"K": n, "N": n, "method": "none", "ratio_target": 1.0},
    )
    prepared = runner.prepare(encoded, selection)
    # Rows that predict the answer tokens: the last prompt position, then answer[:-1].
    answer_rows = torch.arange(prepared.seq_len - 1, prepared.seq_len - 1 + answer_ids.numel())
    q_rows = question_positions(encoded.input_ids, encoded.image_positions, encoded.question_ids)
    rows = torch.cat([answer_rows, q_rows]).to(runner.device)
    cols = encoded.image_positions.to(runner.device)
    captured: list[Tensor] = []

    def hook(_module, _args, output) -> None:
        weights = output[1]
        if weights is None:
            raise RuntimeError("Attention weights are not returned; the eager switch did not reach this layer.")
        w = weights[0][:, rows][:, :, cols].float()  # [H, R, N]
        captured.append((w / w.sum(-1, keepdim=True).clamp_min(1e-12)).mean(0))

    config = runner.language_model.config
    previous = config._attn_implementation
    handles = [decoder_layers[index].self_attn.register_forward_hook(hook) for index in layers]
    try:
        config._attn_implementation = "eager"
        with torch.no_grad():
            runner.teacher_forced_logits(prepared, answer_ids)
    finally:
        config._attn_implementation = previous
        for handle in handles:
            handle.remove()
    per_row = torch.stack(captured).mean(0)  # [R, N]
    answer = per_row[: answer_rows.numel()].mean(0)
    question = per_row[answer_rows.numel() :].mean(0) if q_rows.numel() else answer
    return AttentionTeacher(answer=answer.cpu(), question=question.cpu(), question_rows=int(q_rows.numel()))
