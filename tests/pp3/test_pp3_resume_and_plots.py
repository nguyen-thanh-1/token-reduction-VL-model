import json

import numpy as np

from token_reduction_vl.pruning.pp3_scorer.data import SaliencyRecord, ShardWriter, load_records, stored_sample_ids
from token_reduction_vl.pruning.pp3_scorer.plots import plot_history
from token_reduction_vl.pruning.pp3_scorer.train import RouteBConfig, train_route_b


def make_record(index: int, n: int = 12) -> SaliencyRecord:
    rng = np.random.default_rng(index)
    return SaliencyRecord(
        sample_id=f"s{index}", image_id=f"img{index // 2}", grid_hw=(3, 4),
        v=rng.standard_normal((n, 8)).astype(np.float16), f=rng.standard_normal((3, n, 4)).astype(np.float16),
        z_rel=rng.standard_normal(n).astype(np.float32), z_ms=rng.standard_normal(n).astype(np.float32),
        q=rng.standard_normal((2, 8)).astype(np.float16), saliency=rng.random(n).astype(np.float32),
    )


def test_shard_writer_resumes_numbering_and_reports_stored_ids(tmp_path) -> None:
    first = ShardWriter(tmp_path, "lab_from000000", shard_size=2)
    for index in range(3):
        first.add(make_record(index))
    first.flush()
    second = ShardWriter(tmp_path, "lab_from000000", shard_size=2)  # a resumed run of the same range
    assert second.shard_index == 2
    second.add(make_record(3))
    second.flush()
    assert stored_sample_ids(tmp_path, "lab") == {"s0", "s1", "s2", "s3"}
    assert len(load_records(tmp_path, "lab")) == 4
    assert len(load_records(tmp_path, "lab", max_samples=2)) == 2


def test_training_resumes_from_last_checkpoint_and_plots(tmp_path) -> None:
    records = [make_record(index) for index in range(12)]
    config = RouteBConfig(arch="3a", epochs=2, batch_size=4, patience=10, device="cpu")
    checkpoints, logs = tmp_path / "ckpt", tmp_path / "logs"
    train_route_b(records, (8, 4, 8), config, checkpoints, logs)
    history = json.loads((logs / "history_3a_seed0.json").read_text())
    assert [entry["epoch"] for entry in history] == [0, 1]

    longer = RouteBConfig(arch="3a", epochs=3, batch_size=4, patience=10, device="cpu")
    train_route_b(records, (8, 4, 8), longer, checkpoints, logs, resume=True)
    history = json.loads((logs / "history_3a_seed0.json").read_text())
    assert [entry["epoch"] for entry in history] == [0, 1, 2]  # only epoch 2 was trained again
    assert (checkpoints / "score_head_3a_seed0.pt").exists()
    assert plot_history(logs / "history_3a_seed0.json").exists()
