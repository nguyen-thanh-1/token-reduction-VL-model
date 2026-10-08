"""Read canonical JSONL records for runtime use."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .messages import canonical_to_vqa
from .samples import VQASample
from .schema import CanonicalSample


def iter_canonical_records(path: Path) -> Iterator[CanonicalSample]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            try:
                yield CanonicalSample.from_dict(value)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid canonical record at {path}:{line_number}: {exc}") from exc


def iter_canonical_samples(path: Path, *, project_root: Path) -> Iterator[VQASample]:
    for sample in iter_canonical_records(path):
        yield canonical_to_vqa(sample, project_root=project_root)
