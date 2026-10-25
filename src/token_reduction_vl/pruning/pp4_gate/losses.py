"""Rate–utility objective of PP4 (``method.md`` §5) and its schedule (§6)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def distillation_kl(student_logits: Tensor, teacher_ids: Tensor, teacher_logprobs: Tensor) -> Tensor:
    """Mean over answer positions of ``KL(p_T ‖ p_S)`` on the teacher's stored top-k tokens.

    ``teacher_logprobs`` ``[A, k]`` are renormalized over the stored top-k;
    the student keeps its full-vocabulary normalization.
    """

    teacher = torch.softmax(teacher_logprobs.float(), dim=-1)
    student = torch.log_softmax(student_logits.float(), dim=-1).gather(1, teacher_ids.long())
    return (teacher * (torch.log(teacher.clamp_min(1e-12)) - student)).sum(dim=-1).mean()


def rate_loss(open_prob: Tensor) -> Tensor:
    """``E[K] / N``."""

    return open_prob.sum() / open_prob.numel()


def redundancy_loss(open_prob: Tensor, v_unit: Tensor, threshold: float = 0.8) -> Tensor:
    """``(1/N²) Σ_{i≠j} π_i π_j · max(0, cos(v_i, v_j) − s₀)`` — a differentiable stand-in for DPP."""

    similarity = (v_unit @ v_unit.T - threshold).clamp_min(0.0)
    similarity = similarity - torch.diag(torch.diagonal(similarity))
    n = open_prob.numel()
    return (open_prob.unsqueeze(0) * similarity * open_prob.unsqueeze(1)).sum() / (n * n)


def anchor_loss(open_prob: Tensor, anchor_keep: Tensor) -> Tensor:
    """BCE between ``π`` and the DSFold selection at L3."""

    target = torch.zeros_like(open_prob)
    target[anchor_keep] = 1.0
    return torch.nn.functional.binary_cross_entropy(open_prob.clamp(1e-6, 1 - 1e-6), target)


@dataclass(frozen=True)
class GateSchedule:
    """λ ramps 0 → λ_target over 10–30% of steps; κ decays 1 → 0.5 → 0 over 10–50%."""

    total_steps: int
    lambda_target: float

    def weights(self, step: int) -> tuple[float, float]:
        progress = step / max(1, self.total_steps)
        if progress < 0.1:
            return 0.0, 1.0
        if progress < 0.3:
            ramp = (progress - 0.1) / 0.2
            return self.lambda_target * ramp, 1.0 - 0.5 * ramp
        if progress < 0.5:
            return self.lambda_target, 0.5 * (1.0 - (progress - 0.3) / 0.2)
        return self.lambda_target, 0.0


@dataclass(frozen=True)
class LossTerms:
    total: Tensor
    kl: float
    rate: float
    redundancy: float
    anchor: float
    expected_ratio: float


def gate_objective(
    *,
    student_logits: Tensor,
    teacher_ids: Tensor,
    teacher_logprobs: Tensor,
    open_prob: Tensor,
    v_unit: Tensor,
    anchor_keep: Tensor,
    lam: float,
    kappa: float,
    eta: float,
    mu: float = 0.0,
    gt_ids: Tensor | None = None,
) -> LossTerms:
    kl = distillation_kl(student_logits, teacher_ids, teacher_logprobs)
    rate = rate_loss(open_prob)
    redundancy = redundancy_loss(open_prob, v_unit)
    anchor = anchor_loss(open_prob, anchor_keep)
    total = kl + lam * rate + eta * redundancy + kappa * anchor
    if mu and gt_ids is not None:
        length = min(int(gt_ids.numel()), int(student_logits.shape[0]))
        total = total + mu * torch.nn.functional.cross_entropy(student_logits[:length], gt_ids[:length].long())
    return LossTerms(
        total=total,
        kl=float(kl),
        rate=float(rate),
        redundancy=float(redundancy),
        anchor=float(anchor),
        expected_ratio=float(rate),
    )
