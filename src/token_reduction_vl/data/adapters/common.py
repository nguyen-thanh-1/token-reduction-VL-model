"""Shared helpers for raw-dataset adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from token_reduction_vl.data.samples import iter_saved_rows


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def portable_image_path(dataset_dir: Path, value: Any, project_root: Path) -> str:
    """Resolve a raw image reference and prefer a project-relative path."""

    text = clean_text(value)
    if not text:
        raise ValueError(f"Missing image path in {dataset_dir}.")
    path = Path(text)
    if not path.is_absolute():
        path = dataset_dir / path
    path = path.resolve()
    try:
        return path.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def remaining_metadata(
    row: Mapping[str, Any], *, excluded: Iterable[str], source_index: int
) -> dict[str, Any]:
    blocked = set(excluded)
    return {
        "source_index": source_index,
        **{key: value for key, value in row.items() if key not in blocked},
    }


def rows(path: Path):
    """Expose the common JSONL/legacy-Arrow row iterator to adapters."""

    return iter_saved_rows(path)
