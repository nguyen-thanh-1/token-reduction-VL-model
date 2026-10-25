from pathlib import Path

import torch

from token_reduction_vl.pruning import PrunerInput, validate_output
from token_reduction_vl.pruning.pp3_scorer.dsfold_base import (
    DSFoldBase,
    DSFoldParams,
    fold_deepstack,
    greedy_dpp,
    load_dsfold_params,
    select_order,
    unit_rows,
)
from token_reduction_vl.pruning.pp3_scorer.dsfold_base.scoring import question_embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_input(n: int = 24, dim: int = 16, ratio: float | None = 0.25, seed: int = 0) -> PrunerInput:
    generator = torch.Generator().manual_seed(seed)
    return PrunerInput(
        v=torch.randn(n, dim, generator=generator),
        deepstack=[torch.randn(n, dim, generator=generator) for _ in range(3)],
        grid_hw=(4, n // 4),
        question_ids=torch.tensor([3, 4, 5]),
        question_tokens=("What", " colour", "?"),
        embed_tokens=torch.nn.Embedding(10, dim),
        ratio=ratio,
    )


def test_config_file_matches_default_parameter_set() -> None:
    assert load_dsfold_params(PROJECT_ROOT / "configs" / "dsfold_D-1.0.yaml") == DSFoldParams()


def test_stopwords_are_removed_from_question_tokens() -> None:
    x = make_input()
    q = question_embeddings(x, DSFoldParams().stopwords)
    assert q.shape == (1, 16)  # "What" and "?" are stopwords; " colour" remains
    assert torch.allclose(q[0], x.embed_tokens.weight[4])


def test_greedy_dpp_avoids_duplicates_and_is_nested() -> None:
    base = torch.eye(4)
    v = torch.cat([base, base, base])  # every direction three times
    quality = torch.ones(12)
    result = greedy_dpp(unit_rows(v), quality, 4)
    picked = {int(i) % 4 for i in result.order}
    assert picked == {0, 1, 2, 3}
    longer = greedy_dpp(unit_rows(v), quality, 6)
    assert torch.equal(longer.order[:4], result.order)


def test_dpp_fills_by_quality_when_rank_is_exhausted() -> None:
    v = torch.tensor([[1.0, 0.0]] * 5)
    quality = torch.tensor([1.0, 3.0, 2.0, 0.5, 0.1])
    result = greedy_dpp(unit_rows(v), quality, 3)
    assert result.order.tolist() == [1, 2, 0]


def test_large_images_switch_to_maxmin() -> None:
    v = unit_rows(torch.randn(30, 8, generator=torch.Generator().manual_seed(1)))
    assert select_order(v, torch.ones(30), 5, n_cap=20, eps=1e-6).method == "maxmin"
    assert select_order(v, torch.ones(30), 5, n_cap=64, eps=1e-6).method == "dpp"


def test_fold_preserves_norms_and_gamma_zero_is_identity() -> None:
    x = make_input()
    v_unit = unit_rows(x.v)
    keep = torch.tensor([0, 5, 10])
    drop = torch.tensor([1, 2, 6, 11])
    owner = torch.tensor([0, 0, 1, 2])
    folded = fold_deepstack(list(x.deepstack), v_unit, keep, drop, owner, gamma=0.5, tau=0.1)
    for level, out in zip(x.deepstack, folded, strict=True):
        assert torch.allclose(out.norm(dim=-1), level[keep].norm(dim=-1), atol=1e-5)
        assert not torch.allclose(out, level[keep])
    identity = fold_deepstack(list(x.deepstack), v_unit, keep, drop, owner, gamma=0.0, tau=0.1)
    assert all(torch.equal(a, b[keep]) for a, b in zip(identity, x.deepstack, strict=True))


def test_dsfold_base_output_is_valid_at_every_level() -> None:
    from token_reduction_vl.pruning import BUDGET_LEVELS
    from dataclasses import replace

    x = make_input(n=36)
    for ratio in BUDGET_LEVELS.values():
        xr = replace(x, ratio=ratio)
        out = DSFoldBase().select(xr)
        validate_output(xr, out)
