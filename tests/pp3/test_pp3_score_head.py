import math

import numpy as np
import pytest
import torch

from token_reduction_vl.pruning import PrunerInput, validate_output
from token_reduction_vl.pruning.pp3_scorer import (
    FeatureCompressor,
    ScoredDSFoldPruner,
    build_score_head,
    build_token_features,
    quality_from_scores,
)
from token_reduction_vl.pruning.pp3_scorer.data import SaliencyRecord, ShardWriter, load_records, split_by_image
from token_reduction_vl.pruning.pp3_scorer.dsfold_base import DSFoldBase
from token_reduction_vl.pruning.pp3_scorer.losses import distill_loss, listnet_loss, recall_at_k, spearman
from token_reduction_vl.pruning.pp3_scorer.models import UNIT_DIM

DIM = 32


def make_input(n: int = 20, ratio: float = 2 / 9) -> PrunerInput:
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


def features(x: PrunerInput, compressor=None):
    base = DSFoldBase()
    return build_token_features(x, base.scores(x), stopwords=base.params.stopwords, ratio=x.ratio, compressor=compressor)


def test_unit_dimension_matches_method_doc() -> None:
    assert UNIT_DIM == 467


@pytest.mark.parametrize("arch", ["3a", "3b", "3c", "3d"])
def test_each_architecture_scores_every_token(arch: str) -> None:
    x = make_input()
    head = build_score_head(arch, (DIM, DIM, DIM))
    scores = head(features(x))
    assert scores.shape == (20,)
    assert bool((quality_from_scores(scores) > 0).all())


def test_listnet_prefers_scores_ordered_like_saliency() -> None:
    saliency = torch.tensor([5.0, 1.0, 0.1, 0.1])
    good = listnet_loss(torch.tensor([3.0, 1.0, 0.0, 0.0]), saliency)
    bad = listnet_loss(torch.tensor([0.0, 0.0, 1.0, 3.0]), saliency)
    assert good < bad
    assert recall_at_k(torch.tensor([3.0, 2.0, 1.0, 0.0]), torch.tensor([9.0, 8.0, 0.0, 0.0]), 2) == 1.0
    assert spearman(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([10.0, 20.0, 30.0])) == pytest.approx(1.0)


def test_scored_pruner_returns_valid_output() -> None:
    x = make_input()
    pruner = ScoredDSFoldPruner(build_score_head("3c", (DIM, DIM, DIM)))
    out = pruner.select(x)
    validate_output(x, out)
    assert out.metadata["K"] == math.ceil(20 * 2 / 9 - 1e-6)
    assert out.metadata["method"] == "pp3_scorer"


def test_compressor_and_shards_round_trip(tmp_path) -> None:
    x = make_input()
    base = DSFoldBase()
    scores = base.scores(x)
    compressor = FeatureCompressor.fit(scores.v_unit, scores.f_hat, torch.randn(10, DIM), v_dim=8, level_dim=4, q_dim=8)
    compressor.save(tmp_path / "c.npz")
    loaded = FeatureCompressor.load(tmp_path / "c.npz")
    assert loaded.dims == (8, 4, 8)
    compressed = features(x, loaded)
    assert compressed.v.shape == (20, 8)

    writer = ShardWriter(tmp_path, "sal", shard_size=2)
    for index in range(3):
        writer.add(
            SaliencyRecord(
                sample_id=f"s{index}", image_id=f"img{index // 2}", grid_hw=x.grid_hw,
                v=compressed.v.numpy(), f=torch.stack(compressed.f).numpy(),
                z_rel=compressed.z_rel.numpy(), z_ms=compressed.z_ms.numpy(),
                q=compressed.q.detach().numpy(), saliency=np.arange(20, dtype=np.float32),
            )
        )
    writer.flush()
    records = load_records(tmp_path, "sal")
    assert [r.sample_id for r in records] == ["s0", "s1", "s2"]
    assert records[2].features(0.0).v.shape == (20, 8)
    train, held_out = split_by_image(records, 0.5, seed=0)
    images = lambda idx: {records[i].image_id for i in idx}  # noqa: E731
    assert not images(train) & images(held_out)


def test_listnet_with_signed_saliency_pushes_harmful_tokens_down() -> None:
    signed = torch.tensor([2.0, 0.5, -0.1, -3.0])
    harmful_last = listnet_loss(torch.tensor([3.0, 1.0, 0.0, -2.0]), signed)
    harmful_first = listnet_loss(torch.tensor([1.0, 0.0, -2.0, 3.0]), signed)
    assert harmful_last < harmful_first
    # Non-negative saliency keeps the original normalisation by the mean.
    positive = torch.tensor([5.0, 1.0, 0.1, 0.1])
    expected = -(torch.softmax(positive / positive.mean() / 0.5, 0) * torch.log_softmax(torch.zeros(4), 0)).sum()
    assert listnet_loss(torch.zeros(4), positive) == pytest.approx(float(expected))


def test_text_guided_head_without_question_tokens() -> None:
    x = make_input()
    x = PrunerInput(**{**x.__dict__, "question_ids": torch.zeros(0, dtype=torch.long), "question_tokens": ()})
    scores = build_score_head("3d", (DIM, DIM, DIM))(features(x))
    assert scores.shape == (20,) and bool(torch.isfinite(scores).all())


def test_distill_loss_is_smallest_at_the_teacher_distribution() -> None:
    teacher = torch.tensor([0.6, 0.3, 0.1, 0.0])
    matched = distill_loss(torch.log(teacher.clamp_min(1e-12)), teacher)
    reversed_ = distill_loss(torch.log(teacher.flip(0).clamp_min(1e-12)), teacher)
    assert float(matched) < float(reversed_)
    # T > 1 flattens the target, so a sharply peaked prediction is penalised more.
    peaked = torch.tensor([6.0, 0.0, 0.0, 0.0])
    assert float(distill_loss(peaked, teacher, 4.0)) > float(distill_loss(peaked, teacher, 1.0))


def test_shards_store_a_second_target(tmp_path) -> None:
    def record(name: str) -> SaliencyRecord:
        return SaliencyRecord(
            sample_id=name, image_id=name, grid_hw=(2, 2),
            v=np.zeros((4, 3)), f=np.zeros((3, 4, 2)), z_rel=np.zeros(4), z_ms=np.zeros(4), q=np.zeros((1, 3)),
            saliency=np.array([0.4, 0.3, 0.2, 0.1]), saliency_q=np.array([0.0, 0.0, 0.5, 0.5]),
        )

    writer = ShardWriter(tmp_path, "attn", shard_size=2)
    writer.add(record("a"))
    writer.flush()
    loaded = load_records(tmp_path, "attn")[0]
    assert loaded.target_tensor("question").tolist() == [0.0, 0.0, 0.5, 0.5]
    assert loaded.target_tensor("mix", 0.5).tolist() == pytest.approx([0.2, 0.15, 0.35, 0.3])
    with pytest.raises(ValueError):
        SaliencyRecord(**{**loaded.__dict__, "saliency_q": None}).target_tensor("mix")
