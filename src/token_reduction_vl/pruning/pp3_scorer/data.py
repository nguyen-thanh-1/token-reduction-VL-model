"""Sharded storage of route-B training samples under ``data/features/pp3_scorer/``.

One shard holds up to ``shard_size`` samples. Variable-length per-token arrays
are concatenated and indexed by ``offsets`` (tokens) and ``q_offsets`` (question).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .features import TokenFeatures, fourier_positions


@dataclass(frozen=True)
class SaliencyRecord:
    sample_id: str
    image_id: str
    grid_hw: tuple[int, int]
    v: np.ndarray  # [N, Dv] fp16 (compressed)
    f: np.ndarray  # [3, N, Df] fp16 (compressed)
    z_rel: np.ndarray  # [N]
    z_ms: np.ndarray  # [N]
    q: np.ndarray  # [N_q, Dq] fp16 (compressed)
    saliency: np.ndarray  # [N]

    def features(self, log_ratio: float, device: torch.device | str = "cpu") -> TokenFeatures:
        n = int(self.v.shape[0])
        as_tensor = lambda array: torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)  # noqa: E731
        return TokenFeatures(
            v=as_tensor(self.v),
            f=[as_tensor(level) for level in self.f],
            z_rel=as_tensor(self.z_rel),
            z_ms=as_tensor(self.z_ms),
            positions=fourier_positions(n, self.grid_hw, torch.device(device)),
            q=as_tensor(self.q),
            grid_hw=self.grid_hw,
            log_ratio=log_ratio,
        )

    def saliency_tensor(self, device: torch.device | str = "cpu") -> Tensor:
        return torch.from_numpy(np.asarray(self.saliency, dtype=np.float32)).to(device)


class ShardWriter:
    def __init__(self, directory: Path, prefix: str, shard_size: int = 1000) -> None:
        self.directory = directory
        self.prefix = prefix
        self.shard_size = shard_size
        self.buffer: list[SaliencyRecord] = []
        self.shard_index = 0
        directory.mkdir(parents=True, exist_ok=True)

    def add(self, record: SaliencyRecord) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        records = self.buffer
        offsets = np.cumsum([0, *[r.v.shape[0] for r in records]])
        q_offsets = np.cumsum([0, *[r.q.shape[0] for r in records]])
        np.savez(
            self.directory / f"{self.prefix}_{self.shard_index:04d}.npz",
            v=np.concatenate([r.v for r in records]).astype(np.float16),
            f=np.concatenate([r.f for r in records], axis=1).astype(np.float16),
            z_rel=np.concatenate([r.z_rel for r in records]).astype(np.float32),
            z_ms=np.concatenate([r.z_ms for r in records]).astype(np.float32),
            q=np.concatenate([r.q for r in records]).astype(np.float16),
            saliency=np.concatenate([r.saliency for r in records]).astype(np.float32),
            offsets=offsets,
            q_offsets=q_offsets,
            grid=np.array([r.grid_hw for r in records], dtype=np.int32),
            meta=np.array(json.dumps([[r.sample_id, r.image_id] for r in records])),
        )
        self.shard_index += 1
        self.buffer = []


def iter_shard(path: Path) -> Iterator[SaliencyRecord]:
    data = np.load(path)
    ids = json.loads(str(data["meta"]))
    offsets, q_offsets = data["offsets"], data["q_offsets"]
    for index, (sample_id, image_id) in enumerate(ids):
        start, end = int(offsets[index]), int(offsets[index + 1])
        q_start, q_end = int(q_offsets[index]), int(q_offsets[index + 1])
        yield SaliencyRecord(
            sample_id=sample_id,
            image_id=image_id,
            grid_hw=(int(data["grid"][index][0]), int(data["grid"][index][1])),
            v=data["v"][start:end],
            f=data["f"][:, start:end],
            z_rel=data["z_rel"][start:end],
            z_ms=data["z_ms"][start:end],
            q=data["q"][q_start:q_end],
            saliency=data["saliency"][start:end],
        )


def load_records(directory: Path, prefix: str, max_samples: int | None = None) -> list[SaliencyRecord]:
    """Load shards into memory (≈ 240 KB per GQA sample; 50k samples ≈ 12 GB RAM)."""

    records: list[SaliencyRecord] = []
    for path in sorted(directory.glob(f"{prefix}_*.npz")):
        for record in iter_shard(path):
            records.append(record)
            if max_samples is not None and len(records) >= max_samples:
                return records
    return records


def split_by_image(records: list[SaliencyRecord], holdout_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Indices for train / early-stop with no image on both sides."""

    images = sorted({record.image_id for record in records})
    generator = np.random.default_rng(seed)
    generator.shuffle(images)
    holdout = set(images[: max(1, math.ceil(holdout_fraction * len(images)))])
    train = [i for i, record in enumerate(records) if record.image_id not in holdout]
    early_stop = [i for i, record in enumerate(records) if record.image_id in holdout]
    return train, early_stop
