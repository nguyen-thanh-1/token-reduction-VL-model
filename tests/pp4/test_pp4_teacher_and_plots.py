import json

import numpy as np

from token_reduction_vl.pruning.pp4_gate.plots import plot_training_log
from token_reduction_vl.pruning.pp4_gate.teacher import TeacherShardWriter, load_teacher_targets, stored_teacher_ids


def test_teacher_shards_round_trip_and_resume(tmp_path) -> None:
    writer = TeacherShardWriter(tmp_path, "teacher_from000000", top_k=4, shard_size=2)
    for index, length in enumerate([1, 3, 2]):
        ids = np.arange(length * 4, dtype=np.int32).reshape(length, 4)
        writer.add(f"s{index}", list(range(10, 10 + length)), ids, np.full((length, 4), -1.0, dtype=np.float32))
    writer.flush()
    assert TeacherShardWriter(tmp_path, "teacher_from000000", top_k=4).shard_index == 2
    assert stored_teacher_ids(tmp_path, "teacher") == {"s0", "s1", "s2"}
    targets = load_teacher_targets(tmp_path, "teacher")
    assert targets["s1"].answer_ids.tolist() == [10, 11, 12]
    assert tuple(targets["s1"].topk_ids.shape) == (3, 4)
    assert tuple(targets["s0"].topk_logprobs.shape) == (1, 4)


def test_training_log_plot(tmp_path) -> None:
    log = tmp_path / "lambda1.jsonl"
    rows = [
        {"step": s, "lambda": 0.5, "kappa": 1.0, "total": 2.0 / s, "kl": 1.0 / s, "expected_ratio": 0.5,
         "redundancy": 0.01, "anchor": 0.3}
        for s in (50, 100, 150)
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert plot_training_log(log).exists()
