import torch

from token_reduction_vl.pruning import PrunerInput, validate_output
from token_reduction_vl.pruning.pp3_scorer import build_score_head
from token_reduction_vl.pruning.pp3_scorer.diagnostics import RandomKeepPruner, TopScorePruner
from token_reduction_vl.pruning.pp3_scorer.dsfold_base import DSFoldParams

DIM = 32


def make_input(n: int = 20, ratio: float = 0.75) -> PrunerInput:
    generator = torch.Generator().manual_seed(0)
    return PrunerInput(
        v=torch.randn(n, DIM, generator=generator),
        deepstack=[torch.randn(n, DIM, generator=generator) for _ in range(3)],
        grid_hw=(4, n // 4),
        question_ids=torch.tensor([1, 2]),
        question_tokens=(" dog", " colour"),
        embed_tokens=torch.nn.Embedding(5, DIM),
        ratio=ratio,
    )


def test_random_pruner_is_valid_and_reproducible() -> None:
    x = make_input()
    first = RandomKeepPruner(seed=7).select(x)
    second = RandomKeepPruner(seed=7).select(x)
    validate_output(x, first)
    assert first.metadata["K"] == 15
    assert torch.equal(first.keep_indices, second.keep_indices)


def test_tiny_ratio_keeps_one_token() -> None:
    x = make_input(ratio=1e-6)
    out = RandomKeepPruner().select(x)
    validate_output(x, out)
    assert out.metadata["K"] == 1


def test_top_score_pruner_drops_the_lowest_quality_tokens() -> None:
    x = make_input(ratio=0.9)
    pruner = TopScorePruner(build_score_head("3a", (DIM, DIM, DIM)), params=DSFoldParams().with_overrides(gamma=0.0))
    out = pruner.select(x)
    validate_output(x, out)
    _, quality = pruner.token_quality(x)
    dropped = sorted(set(range(20)) - set(out.keep_indices.tolist()))
    assert len(dropped) == 2
    assert set(dropped) == set(torch.topk(quality, 2, largest=False).indices.tolist())
    # gamma = 0: kept DeepStack features are passed through unchanged.
    assert torch.allclose(out.deepstack_kept[0], x.deepstack[0][out.keep_indices])


def test_most_negative_only_drops_negative_entries() -> None:
    from token_reduction_vl.pruning.pp3_scorer.diagnostics import complement, most_negative

    saliency = torch.tensor([0.5, -2.0, 0.1, -0.1, -1.0, 3.0, 0.2, 0.3, 0.4, 0.6])
    assert sorted(most_negative(saliency, 0.2).tolist()) == [1, 4]
    assert sorted(most_negative(saliency, 0.9).tolist()) == [1, 3, 4]
    assert most_negative(torch.ones(4), 0.5).numel() == 0
    assert complement(5, torch.tensor([0, 3])).tolist() == [1, 2, 4]


def test_keep_indices_pruner_returns_the_given_set() -> None:
    from token_reduction_vl.pruning.pp3_scorer.diagnostics import KeepIndicesPruner

    x = make_input(ratio=None)
    out = KeepIndicesPruner(torch.tensor([7, 2, 11]), DSFoldParams().with_overrides(gamma=0.0)).select(x)
    validate_output(x, out)
    assert out.keep_indices.tolist() == [2, 7, 11]
