"""PP1 conformance (pp1-dsfold/method.md §13): every member's DSFold copy must agree.

The synthetic part always runs on CPU. The fixture part runs once the shared
tensors exist in data/features/pp1_conformance/ (not tracked by Git).
"""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from token_reduction_vl.pruning import BUDGET_LEVELS, PrunerInput
from token_reduction_vl.pruning.pp3_scorer.dsfold_base import DSFoldBase as PP3Base
from token_reduction_vl.pruning.pp4_gate.dsfold_base import DSFoldBase as PP4Base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "data" / "features" / "pp1_conformance"
IMPLEMENTATIONS = {"pp3_scorer": PP3Base, "pp4_gate": PP4Base}


def synthetic_input(n: int, seed: int, dim: int = 64) -> PrunerInput:
    generator = torch.Generator().manual_seed(seed)
    width = 8
    return PrunerInput(
        v=torch.randn(n, dim, generator=generator),
        deepstack=[torch.randn(n, dim, generator=generator) for _ in range(3)],
        grid_hw=(n // width, width),
        question_ids=torch.tensor([1, 2, 3]),
        question_tokens=("Is", " the", " sky"),
        embed_tokens=torch.nn.Embedding(8, dim),
        ratio=None,
    )


@pytest.mark.parametrize("n,seed", [(64, 0), (96, 1), (160, 2)])
@pytest.mark.parametrize("level", ["L1", "L3", "L5"])
def test_all_dsfold_copies_agree_on_synthetic_inputs(n: int, seed: int, level: str) -> None:
    x = replace(synthetic_input(n, seed), ratio=BUDGET_LEVELS[level])
    outputs = {name: impl().select(x) for name, impl in IMPLEMENTATIONS.items()}
    reference = outputs["pp3_scorer"]
    for name, out in outputs.items():
        assert torch.equal(out.keep_indices, reference.keep_indices), name
        for a, b in zip(out.deepstack_kept, reference.deepstack_kept, strict=True):
            assert torch.allclose(a, b, atol=1e-3), name


def test_all_dsfold_copies_agree_on_the_maxmin_branch() -> None:
    x = replace(synthetic_input(160, 3), ratio=BUDGET_LEVELS["L3"])
    outputs = [impl(impl().params.with_overrides(n_cap=100)).select(x) for impl in IMPLEMENTATIONS.values()]
    assert all(out.metadata["selector"] == "maxmin" for out in outputs)
    assert torch.equal(outputs[0].keep_indices, outputs[1].keep_indices)


REFERENCE = PROJECT_ROOT / "tests" / "fixtures" / "pp1_conformance" / "reference.json"


def test_fixture_round_trip(tmp_path) -> None:
    from token_reduction_vl.pruning.conformance import load_fixture, save_fixture

    x = replace(synthetic_input(64, 7), ratio=None)
    save_fixture(tmp_path / "s.npz", x, sample_id="gqa:testdev:1")
    sample_id, loaded = load_fixture(tmp_path / "s.npz", BUDGET_LEVELS["L3"])
    assert sample_id == "gqa:testdev:1"
    assert loaded.grid_hw == x.grid_hw and loaded.question_tokens == x.question_tokens
    assert torch.allclose(loaded.embed_tokens.weight, x.embed_tokens.weight[x.question_ids], atol=1e-6)


@pytest.mark.skipif(not (FIXTURES.exists() and REFERENCE.exists()), reason="shared conformance fixtures not generated yet")
def test_fixture_reference_matches() -> None:  # pragma: no cover - needs data/features
    import json

    from token_reduction_vl.pruning.conformance import load_fixture

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    for entry in reference["entries"]:
        _, x = load_fixture(FIXTURES / entry["file"], BUDGET_LEVELS[entry["level"]])
        for name, impl in IMPLEMENTATIONS.items():
            out = impl().select(x)
            assert out.keep_indices.tolist() == entry["keep_indices"], (name, entry["file"], entry["level"])
            sums = [float(level.double().sum()) for level in out.deepstack_kept]
            assert sums == pytest.approx(entry["deepstack_sum"], rel=1e-4, abs=1e-2), (name, entry["file"])
