import numpy as np
import pytest
import torch

from token_reduction_vl.pruning import PrunerInput, validate_output
from token_reduction_vl.pruning.pp3_scorer import build_score_head
from token_reduction_vl.pruning.pp3_scorer.data import SaliencyRecord, contrastive_records
from token_reduction_vl.pruning.pp3_scorer.focus_context import (
    FocusContextParams,
    FocusContextPruner,
    context_tokens,
    focus_share,
    grow_regions,
    neighbor_matrix,
)
from token_reduction_vl.reporting.region_metrics import components

GRID = (4, 6)
DIM = 32


def make_input(n: int = 24, ratio: float = 1 / 3) -> PrunerInput:
    generator = torch.Generator().manual_seed(0)
    return PrunerInput(
        v=torch.randn(n, DIM, generator=generator),
        deepstack=[torch.randn(n, DIM, generator=generator) for _ in range(3)],
        grid_hw=GRID,
        question_ids=torch.tensor([1, 2]),
        question_tokens=(" dog", " colour"),
        embed_tokens=torch.nn.Embedding(5, DIM),
        ratio=ratio,
    )


def test_neighbor_matrix_is_the_8_neighbourhood() -> None:
    neighbors = neighbor_matrix(GRID)
    assert int(neighbors[0].sum()) == 3  # corner
    assert int(neighbors[7].sum()) == 8  # interior (1, 1)
    assert not bool(neighbors.diagonal().any())


def test_beta_zero_is_top_k_and_large_beta_grows_one_region() -> None:
    scores = torch.zeros(24)
    scores[0], scores[23], scores[22] = 5.0, 4.9, 4.8
    assert set(grow_regions(scores, 3, GRID, beta=0.0).tolist()) == {0, 22, 23}
    grown = grow_regions(scores, 4, GRID, beta=10.0)
    assert 0 in grown.tolist() and components(grown.tolist(), GRID) == 1


def test_context_takes_one_token_per_cell_and_skips_focus_cells() -> None:
    quality = torch.arange(24, dtype=torch.float)
    exclude = torch.tensor([23])
    chosen = context_tokens(quality, GRID, 3, exclude)
    assert chosen.numel() == 3 and 23 not in chosen.tolist()
    block = 2  # floor(sqrt(24 / 3)) = 2 -> cells of 2x2
    cells = {(i // 6 // block, (i % 6) // block) for i in chosen.tolist()}
    assert len(cells) == 3 and (1, 2) not in cells  # the cell of token 23 holds focus
    assert context_tokens(quality, GRID, 0, exclude).numel() == 0


def test_focus_share_follows_peakedness() -> None:
    params = FocusContextParams(share_min=0.2, share_max=0.9)
    assert focus_share(torch.zeros(50), params) == pytest.approx(0.2)
    peaked = torch.zeros(50)
    peaked[3] = 30.0
    assert focus_share(peaked, params) == pytest.approx(0.9, abs=1e-3)


def test_pruner_returns_valid_output_with_roles() -> None:
    x = make_input(n=24, ratio=1 / 3)
    torch.manual_seed(0)
    pruner = FocusContextPruner(build_score_head("3d", (DIM, DIM, DIM)), build_score_head("3d", (DIM, DIM, DIM)))
    output = pruner.select(x)
    validate_output(x, output)
    focus = output.metadata["focus_indices"]
    assert output.keep_indices.numel() == 8 and 1 <= len(focus) <= 8
    assert set(focus) <= set(output.keep_indices.tolist())
    dpp_output = pruner.with_focus_params(context_dpp=1, share_min=0.25, share_max=0.25).select(x)
    validate_output(x, dpp_output)
    assert dpp_output.keep_indices.numel() == 8 and len(dpp_output.metadata["focus_indices"]) == 2
    assert pruner.with_focus_params(beta=0).focus_params.beta == 0.0
    with pytest.raises(ValueError):
        pruner.with_focus_params(unknown=1)


def test_contrastive_records_remove_the_shared_part() -> None:
    def record(name: str, image: str, saliency: list[float]) -> SaliencyRecord:
        return SaliencyRecord(
            sample_id=name, image_id=image, grid_hw=(2, 2), v=np.zeros((4, 3)), f=np.zeros((3, 4, 2)),
            z_rel=np.zeros(4), z_ms=np.zeros(4), q=np.zeros((1, 3)), saliency=np.array(saliency),
        )

    records = [
        record("a", "img", [0.5, 0.3, 0.2, 0.0]),  # sink on token 0 shared by both questions
        record("b", "img", [0.5, 0.0, 0.2, 0.3]),
        record("c", "alone", [1.0, 0.0, 0.0, 0.0]),
    ]
    result = {r.sample_id: r.saliency for r in contrastive_records(records, label="primary")}
    assert set(result) == {"a", "b"}
    assert result["a"].tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert result["b"].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])
