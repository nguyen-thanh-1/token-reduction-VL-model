"""One PP4 training step through the frozen Qwen3-VL (``method.md`` §4–6).

The LLM sees the *full* visual sequence; gates act through the additive
attention bias and by scaling DeepStack, so the graph stays differentiable.
Fold is computed from the hard set ``{i : z_i > 0}`` and detached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from token_reduction_vl.pruning.base import BUDGET_LEVELS, PrunerOutput, budget_for_ratio

from .attention_bias import gated_attention_mask
from .dsfold_base import DSFoldBase
from .features import build_gate_features
from .hard_concrete import open_probability, sample_gates
from .losses import LossTerms, gate_objective
from .model import GateNet

if TYPE_CHECKING:
    from token_reduction_vl.models.qwen3_vl_pruned import EncodedSample, PrunedQwen3VL


@dataclass(frozen=True)
class GateTrainConfig:
    lambda_target: float = 1.0
    eta: float = 0.1
    mu: float = 0.0
    epochs: int = 2
    lr: float = 1e-4
    grad_accumulation: int = 8
    eps: float = 1e-4
    anchor_level: str = "L3"
    seed: int = 0


@dataclass(frozen=True)
class TeacherTarget:
    answer_ids: Tensor  # [A] teacher greedy answer, no EOS
    topk_ids: Tensor  # [A, k]
    topk_logprobs: Tensor  # [A, k]


def gated_deepstack(base: DSFoldBase, x, scores, gates: Tensor) -> list[Tensor]:
    """Full-length DeepStack ``[N, D]`` where open tokens carry folded features, scaled by ``z``."""

    hard_keep = torch.nonzero(gates > 0, as_tuple=False).flatten()
    if hard_keep.numel() == 0:
        hard_keep = gates.argmax().view(1)
    with torch.no_grad():
        folded = base.fold(x, scores, hard_keep)
    out = []
    for level, folded_level in zip(x.deepstack, folded, strict=True):
        full = level.detach().clone()
        full[hard_keep] = folded_level.to(full.dtype)
        out.append(full * gates.to(full.dtype).unsqueeze(-1))
    return out


def train_step(
    runner: PrunedQwen3VL,
    net: GateNet,
    base: DSFoldBase,
    encoded: EncodedSample,
    teacher: TeacherTarget,
    *,
    lam: float,
    kappa: float,
    config: GateTrainConfig,
) -> LossTerms:
    """Forward + loss for one sample; the caller runs ``backward`` / ``optimizer.step``."""

    x = runner.pruner_input(encoded, None)
    with torch.no_grad():
        scores = base.scores(x)
        anchor = base.order(scores, budget_for_ratio(BUDGET_LEVELS[config.anchor_level], x.num_tokens))
    features = build_gate_features(x, scores, stopwords=base.params.stopwords)
    log_alpha = net(features)
    gates = sample_gates(log_alpha)
    open_prob = open_probability(log_alpha)

    n = x.num_tokens
    selection = PrunerOutput(
        keep_indices=torch.arange(n, device=x.v.device),
        v_kept=x.v.detach(),
        deepstack_kept=gated_deepstack(base, x, scores, gates),
        metadata={"K": n, "N": n, "method": "pp4_gate_train", "ratio_target": None},
    )
    prepared = runner.prepare(encoded, selection)
    total_len = prepared.seq_len + int(teacher.answer_ids.numel())
    mask = gated_attention_mask(
        total_len, encoded.image_positions, gates, eps=config.eps, dtype=prepared.inputs_embeds.dtype
    )
    logits = runner.teacher_forced_logits(prepared, teacher.answer_ids, attention_mask=mask)
    return gate_objective(
        student_logits=logits,
        teacher_ids=teacher.topk_ids.to(logits.device),
        teacher_logprobs=teacher.topk_logprobs.to(logits.device),
        open_prob=open_prob,
        v_unit=scores.v_unit,
        anchor_keep=anchor.order,
        lam=lam,
        kappa=kappa,
        eta=config.eta,
        mu=config.mu,
    )


def save_gate_net(net: GateNet, path, *, config: GateTrainConfig, step: int) -> None:
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "config": asdict(config), "step": step}, path)


def load_gate_net(path, device: str = "cpu") -> GateNet:
    state = torch.load(path, map_location=device, weights_only=False)
    net = GateNet()
    net.load_state_dict(state["state_dict"])
    return net.to(device).eval()
