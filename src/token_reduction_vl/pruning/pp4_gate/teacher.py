"""Sharded storage of full-token teacher targets (``data/features/pp4_gate/``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .train import TeacherTarget

MAX_ANSWER = 8


class TeacherShardWriter:
    def __init__(self, directory: Path, prefix: str, top_k: int, shard_size: int = 500) -> None:
        self.directory = directory
        self.prefix = prefix
        self.top_k = top_k
        self.shard_size = shard_size
        self.rows: list[tuple[str, list[int], np.ndarray, np.ndarray]] = []
        directory.mkdir(parents=True, exist_ok=True)
        self.shard_index = len(list(directory.glob(f"{prefix}_[0-9][0-9][0-9][0-9].npz")))

    def add(self, sample_id: str, answer: list[int], topk_ids: np.ndarray, topk_logprobs: np.ndarray) -> None:
        self.rows.append((sample_id, answer, topk_ids, topk_logprobs))
        if len(self.rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        count = len(self.rows)
        answers = np.full((count, MAX_ANSWER), -1, dtype=np.int32)
        ids = np.full((count, MAX_ANSWER, self.top_k), -1, dtype=np.int32)
        logprobs = np.full((count, MAX_ANSWER, self.top_k), -np.inf, dtype=np.float16)
        for row, (_, answer, topk_ids, topk_logprobs) in enumerate(self.rows):
            length = len(answer)
            answers[row, :length] = answer
            ids[row, :length] = topk_ids
            logprobs[row, :length] = topk_logprobs
        np.savez(
            self.directory / f"{self.prefix}_{self.shard_index:04d}.npz",
            sample_ids=np.array(json.dumps([row[0] for row in self.rows])),
            answers=answers,
            topk_ids=ids,
            topk_logprobs=logprobs,
        )
        self.shard_index += 1
        self.rows = []


def stored_teacher_ids(directory: Path, prefix: str) -> set[str]:
    done: set[str] = set()
    for path in directory.glob(f"{prefix}_*.npz"):
        with np.load(path) as data:
            done.update(json.loads(str(data["sample_ids"])))
    return done


def load_teacher_targets(directory: Path, prefix: str) -> dict[str, TeacherTarget]:
    targets: dict[str, TeacherTarget] = {}
    for path in sorted(directory.glob(f"{prefix}_*.npz")):
        with np.load(path) as data:
            for index, sample_id in enumerate(json.loads(str(data["sample_ids"]))):
                answer = data["answers"][index]
                length = int((answer >= 0).sum())
                targets[sample_id] = TeacherTarget(
                    answer_ids=torch.from_numpy(answer[:length].astype(np.int64)),
                    topk_ids=torch.from_numpy(data["topk_ids"][index][:length].astype(np.int64)),
                    topk_logprobs=torch.from_numpy(data["topk_logprobs"][index][:length].astype(np.float32)),
                )
    return targets
