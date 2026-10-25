"""Fixed sample-id lists under ``configs/splits/`` (``METHODS.md`` §9.3)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from .canonical import iter_canonical_records
from .messages import canonical_to_vqa
from .samples import VQASample


def read_split(path: Path) -> list[str]:
    """One ``sample_id`` per line; lines starting with ``#`` are comments."""

    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def write_split(path: Path, sample_ids: Iterable[str], *, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(sample_ids)
    path.write_text(f"# {comment}\n{body}\n", encoding="utf-8")


def iter_split_samples(canonical_path: Path, sample_ids: Iterable[str], *, project_root: Path) -> Iterator[VQASample]:
    """Stream canonical records, yielding only listed ids (file order, images loaded lazily)."""

    wanted = set(sample_ids)
    for record in iter_canonical_records(canonical_path):
        if record.sample_id in wanted:
            yield canonical_to_vqa(record, project_root=project_root)
