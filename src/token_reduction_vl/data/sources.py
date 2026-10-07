"""Resolve configured dataset sources into canonical samples."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .samples import (
    DEFAULT_ANSWER_FIELDS,
    DEFAULT_ID_FIELDS,
    DEFAULT_IMAGE_FIELDS,
    DEFAULT_QUESTION_FIELDS,
    VQASample,
    iter_paired_samples,
    iter_single_samples,
)


def _fields(source: Mapping[str, Any], name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    values = source.get(name, default)
    return tuple(str(value) for value in values)


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def iter_source_samples(
    source: Mapping[str, Any], *, project_root: Path
) -> Iterable[VQASample]:
    """Build an iterator for one ``data.sources`` entry in the YAML config."""

    common = {
        "image_fields": _fields(source, "image_fields", DEFAULT_IMAGE_FIELDS),
        "question_fields": _fields(source, "question_fields", DEFAULT_QUESTION_FIELDS),
        "answer_fields": _fields(source, "answer_fields", DEFAULT_ANSWER_FIELDS),
        "id_fields": _fields(source, "id_fields", DEFAULT_ID_FIELDS),
    }
    kind = source.get("kind", "single")
    if kind == "single":
        return iter_single_samples(_resolve_path(project_root, source["path"]), **common)
    if kind == "paired":
        join_fields = _fields(source, "join_key_fields", ("image_id", "id"))
        return iter_paired_samples(
            _resolve_path(project_root, source["instructions"]),
            _resolve_path(project_root, source["images"]),
            join_key_fields=join_fields,
            **common,
        )
    raise ValueError(f"Unsupported dataset source kind: {kind!r}")
