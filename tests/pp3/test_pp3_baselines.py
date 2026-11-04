import torch

from token_reduction_vl.pruning import PrunerInput, validate_output
from token_reduction_vl.pruning.pp3_scorer.baselines import (
    CDPrunerPruner,
    DivPrunePruner,
    FastVPruner,
    cdpruner_relevance,
    divprune_order,
)

DIM = 16


def make_input(v: torch.Tensor, ratio: float) -> PrunerInput:
    n = v.shape[0]
    generator = torch.Generator().manual_seed(0)
    return PrunerInput(
        v=v,
        deepstack=[torch.randn(n, DIM, generator=generator) for _ in range(3)],
        grid_hw=(2, n // 2),
        question_ids=torch.tensor([1]),
        question_tokens=(" dog",),
        embed_tokens=torch.nn.Embedding(3, DIM),
        ratio=ratio,
    )


def test_divprune_picks_one_token_per_cluster() -> None:
    generator = torch.Generator().manual_seed(1)
    centers = torch.eye(DIM)[:4] * 10
    v = torch.cat([centers[i] + 0.01 * torch.randn(3, DIM, generator=generator) for i in range(4)])  # 4 clusters x 3
    order = divprune_order(v, 4)
    assert sorted(int(i) // 3 for i in order) == [0, 1, 2, 3]


def test_divprune_output_is_valid_without_fold() -> None:
    x = make_input(torch.randn(12, DIM, generator=torch.Generator().manual_seed(2)), ratio=1 / 3)
    out = DivPrunePruner().select(x)
    validate_output(x, out)
    assert out.metadata["K"] == 4 and out.metadata["method"] == "divprune"
    assert torch.equal(out.deepstack_kept[0], x.deepstack[0][out.keep_indices])  # gamma = 0: nothing folded


def test_fastv_needs_prepared_attention_and_keeps_the_top() -> None:
    x = make_input(torch.randn(12, DIM), ratio=0.25)
    pruner = FastVPruner()
    try:
        pruner.select(x)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    pruner.attention = torch.arange(12, dtype=torch.float)
    out = pruner.select(x)
    validate_output(x, out)
    assert out.keep_indices.tolist() == [9, 10, 11]


def test_cdpruner_relevance_is_min_max_normalised_cosine_to_the_question() -> None:
    x = make_input(torch.randn(12, DIM, generator=torch.Generator().manual_seed(3)), ratio=1 / 3)
    q = x.embed_tokens.weight[1].detach()
    expected = torch.nn.functional.normalize(x.v, dim=-1) @ torch.nn.functional.normalize(q, dim=0)
    expected = (expected - expected.min()) / (expected.max() - expected.min())
    assert torch.allclose(cdpruner_relevance(x), expected, atol=1e-6)


def test_cdpruner_keeps_question_relevant_tokens_without_fold() -> None:
    # Orthogonal tokens (S = I), so the DPP keeps the K most relevant ones.
    x = make_input(torch.eye(DIM)[:12].clone(), ratio=0.25)
    with torch.no_grad():
        x.embed_tokens.weight[1] = 0.0
        x.embed_tokens.weight[1, [2, 7, 11]] = torch.tensor([3.0, 2.0, 1.0])
    out = CDPrunerPruner().select(x)
    validate_output(x, out)
    assert out.metadata["method"] == "cdpruner" and out.keep_indices.tolist() == [2, 7, 11]
    assert torch.equal(out.deepstack_kept[0], x.deepstack[0][out.keep_indices])
