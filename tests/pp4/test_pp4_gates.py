from dataclasses import replace

import pytest
import torch

from token_reduction_vl.pruning import PrunerInput, validate_output
from token_reduction_vl.pruning.pp4_gate import (
    GatedDSFoldPruner,
    GateNet,
    GateSchedule,
    build_gate_features,
    deterministic_gates,
    gated_attention_mask,
    keep_from_gates,
    open_probability,
    sample_gates,
)
from token_reduction_vl.pruning.pp4_gate.dsfold_base import DSFoldBase
from token_reduction_vl.pruning.pp4_gate.hard_concrete import initial_log_alpha
from token_reduction_vl.pruning.pp4_gate.losses import distillation_kl, rate_loss, redundancy_loss
from token_reduction_vl.pruning.pp4_gate.model import UNIT_DIM
from token_reduction_vl.pruning.pp4_gate.train import gated_deepstack

DIM = 32


def make_input(n: int = 36) -> PrunerInput:
    generator = torch.Generator().manual_seed(0)
    return PrunerInput(
        v=torch.randn(n, DIM, generator=generator),
        deepstack=[torch.randn(n, DIM, generator=generator) for _ in range(3)],
        grid_hw=(6, n // 6),
        question_ids=torch.tensor([1]),
        question_tokens=(" sign",),
        embed_tokens=torch.nn.Embedding(4, DIM),
        ratio=None,
    )


def test_initial_open_probability_is_one_half() -> None:
    assert initial_log_alpha() == pytest.approx(-1.599, abs=1e-3)
    assert float(open_probability(torch.tensor(initial_log_alpha()))) == pytest.approx(0.5)


def test_samples_live_in_unit_interval_with_exact_zeros_and_ones() -> None:
    z = sample_gates(torch.zeros(20_000), generator=torch.Generator().manual_seed(0))
    assert float(z.min()) == 0.0 and float(z.max()) == 1.0
    assert 0.0 < float((z == 0).float().mean()) < 0.5


def test_keep_from_gates_respects_threshold_and_minimum() -> None:
    log_alpha = torch.tensor([3.0, -3.0, 2.0, -3.0, -3.0, -3.0])
    assert deterministic_gates(log_alpha)[0] > 0.5
    assert keep_from_gates(log_alpha, min_ratio=0.1).tolist() == [0, 2]
    assert keep_from_gates(torch.full((36,), -5.0), min_ratio=1 / 18).numel() == 2


def test_gate_net_starts_with_uniform_log_alpha() -> None:
    x = make_input()
    base = DSFoldBase()
    net = GateNet(input_dim=DIM)
    log_alpha = net(build_gate_features(x, base.scores(x), stopwords=base.params.stopwords))
    assert UNIT_DIM == 466
    assert log_alpha.shape == (36,)
    assert torch.allclose(log_alpha, torch.full((36,), initial_log_alpha()))


def test_attention_mask_is_causal_with_column_bias() -> None:
    gates = torch.tensor([1.0, 0.0])
    mask = gated_attention_mask(5, torch.tensor([1, 2]), gates, eps=1e-4)
    assert mask.shape == (1, 1, 5, 5)
    blocked = torch.finfo(torch.float32).min
    assert mask[0, 0, 0, 1] == blocked  # future key stays blocked
    assert mask[0, 0, 4, 1] == pytest.approx(torch.log(torch.tensor(1.0001)).item(), abs=1e-6)
    assert mask[0, 0, 4, 2] == pytest.approx(torch.log(torch.tensor(1e-4)).item(), rel=1e-4)
    assert mask[0, 0, 4, 3] == 0.0  # text column untouched


def test_losses_behave_as_documented() -> None:
    teacher_ids = torch.tensor([[2, 0]])
    teacher_lp = torch.log(torch.tensor([[0.75, 0.25]]))
    student = torch.full((1, 4), -1e4)
    student[0, 2], student[0, 0] = torch.log(torch.tensor(0.75)), torch.log(torch.tensor(0.25))
    assert float(distillation_kl(student, teacher_ids, teacher_lp)) == pytest.approx(0.0, abs=1e-5)
    assert float(rate_loss(torch.tensor([1.0, 0.0, 0.5, 0.5]))) == 0.5
    assert float(redundancy_loss(torch.ones(3), torch.eye(3))) == 0.0


def test_schedule_follows_table() -> None:
    schedule = GateSchedule(total_steps=100, lambda_target=2.0)
    assert schedule.weights(5) == (0.0, 1.0)
    assert schedule.weights(20) == pytest.approx((1.0, 0.75))
    assert schedule.weights(40) == pytest.approx((2.0, 0.25))
    assert schedule.weights(80) == (2.0, 0.0)


def test_gated_deepstack_is_differentiable_in_the_gates() -> None:
    x = make_input()
    base = DSFoldBase()
    scores = base.scores(x)
    gates = torch.rand(36).requires_grad_(True)
    levels = gated_deepstack(base, x, scores, gates)
    assert [level.shape for level in levels] == [(36, DIM)] * 3
    sum(level.sum() for level in levels).backward()
    assert gates.grad is not None and float(gates.grad.abs().sum()) > 0


def test_gated_pruner_returns_valid_adaptive_output() -> None:
    x = make_input()
    out = GatedDSFoldPruner(GateNet(input_dim=DIM)).select(x)
    validate_output(x, out)
    assert out.metadata["ratio_target"] is None
    assert out.metadata["K"] >= 2  # K_min = ceil(36 / 18)
    with pytest.raises(ValueError):
        DSFoldBase().select(replace(x, ratio=None))
