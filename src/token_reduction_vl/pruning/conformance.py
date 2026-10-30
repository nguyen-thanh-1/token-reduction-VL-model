"""Shared fixture format for the PP1 conformance test (``pp1-dsfold/method.md`` §13).

A fixture stores one encoded sample as fp16 tensors. Question ids are remapped
to ``0..N_q-1`` and only their embedding rows are kept, so a fixture is small
and self-contained. References are computed from the *reloaded* tensors, so
every member's DSFold copy sees exactly the same input.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .base import PrunerInput


def save_fixture(path: Path, x: PrunerInput, *, sample_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = x.embed_tokens.weight[x.question_ids.to(x.embed_tokens.weight.device)].detach().float().cpu()
    np.savez(
        path,
        sample_id=np.array(sample_id),
        v=x.v.detach().cpu().half().numpy(),
        **{f"f{i}": level.detach().cpu().half().numpy() for i, level in enumerate(x.deepstack)},
        grid_hw=np.array(x.grid_hw, dtype=np.int64),
        question_ids=np.arange(int(x.question_ids.numel()), dtype=np.int64),
        question_tokens=np.array(json.dumps(list(x.question_tokens))),
        question_embeddings=embeddings.numpy().reshape(-1, x.v.shape[1]),
    )


def load_fixture(path: Path, ratio: float | None) -> tuple[str, PrunerInput]:
    with np.load(path) as data:
        embeddings = torch.from_numpy(data["question_embeddings"]).float()
        if embeddings.shape[0] == 0:
            embeddings = torch.zeros((1, int(data["v"].shape[1])))
        x = PrunerInput(
            v=torch.from_numpy(data["v"]).float(),
            deepstack=[torch.from_numpy(data[f"f{i}"]).float() for i in range(3)],
            grid_hw=(int(data["grid_hw"][0]), int(data["grid_hw"][1])),
            question_ids=torch.from_numpy(data["question_ids"]).long(),
            question_tokens=tuple(json.loads(str(data["question_tokens"]))),
            embed_tokens=torch.nn.Embedding.from_pretrained(embeddings),
            ratio=ratio,
        )
        return str(data["sample_id"]), x
