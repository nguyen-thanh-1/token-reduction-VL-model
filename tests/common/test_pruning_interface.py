import pytest
import torch

from token_reduction_vl.pruning import BUDGET_LEVELS, NoOpPruner, PrunerInput, PrunerOutput, budget_for_ratio, validate_output


def make_input(n: int = 12, dim: int = 8, ratio: float | None = 0.5) -> PrunerInput:
    torch.manual_seed(0)
    return PrunerInput(
        v=torch.randn(n, dim),
        deepstack=[torch.randn(n, dim) for _ in range(3)],
        grid_hw=(3, n // 3),
        question_ids=torch.tensor([1, 2]),
        question_tokens=("red", " car"),
        embed_tokens=torch.nn.Embedding(10, dim),
        ratio=ratio,
    )


def test_budget_levels_match_documented_token_counts() -> None:
    # 00-common §6.2: an MME median image (N = 384) keeps 22/43/86/128/192/288/384 tokens.
    assert [budget_for_ratio(r, 384) for r in BUDGET_LEVELS.values()] == [22, 43, 86, 128, 192, 288, 384]
    assert budget_for_ratio(2 / 9, 260) == 58
    assert budget_for_ratio(1 / 18, 3) == 1


def test_budget_rejects_invalid_ratio() -> None:
    with pytest.raises(ValueError):
        budget_for_ratio(0.0, 10)


def test_noop_pruner_satisfies_contract() -> None:
    x = make_input()
    out = NoOpPruner().select(x)
    validate_output(x, out)
    assert torch.equal(out.keep_indices, torch.arange(12))


def test_validate_output_rejects_unsorted_or_mismatched_outputs() -> None:
    x = make_input()
    keep = torch.tensor([3, 1])
    bad = PrunerOutput(keep, x.v[keep], [f[keep] for f in x.deepstack], {"K": 2, "N": 12, "method": "t", "ratio_target": 0.2})
    with pytest.raises(ValueError, match="increasing"):
        validate_output(x, bad)
    keep = torch.tensor([1, 3])
    short = PrunerOutput(keep, x.v[keep], [f[keep] for f in x.deepstack[:2]], {"K": 2, "N": 12, "method": "t", "ratio_target": 0.2})
    with pytest.raises(ValueError, match="DeepStack"):
        validate_output(x, short)
