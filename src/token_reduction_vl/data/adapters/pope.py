"""POPE object-hallucination adapter (ADR 0006)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from token_reduction_vl.data.schema import CanonicalSample

from .common import clean_text, portable_image_path, remaining_metadata, rows

POPE_CATEGORIES = frozenset({"adversarial", "popular", "random"})


def iter_pope_samples(
    dataset_dir: Path, *, split: str, project_root: Path
) -> Iterator[CanonicalSample]:
    """Yes/no questions; ``category`` is the negative-sampling setting."""

    for index, row in enumerate(rows(dataset_dir)):
        category = clean_text(row.get("category")).lower()
        if category not in POPE_CATEGORIES:
            raise ValueError(f"POPE row {index} has unknown category {category!r}.")
        question_id = clean_text(row.get("question_id")) or clean_text(row.get("id")) or str(index)
        image_id = clean_text(row.get("image_source"))
        if not image_id:
            raise ValueError(f"POPE row {index} has no image_source.")
        answer = clean_text(row.get("answer")).lower()
        if answer and answer not in {"yes", "no"}:
            raise ValueError(f"POPE row {index} has a non yes/no answer {answer!r}.")
        yield CanonicalSample(
            sample_id=f"pope:{split}:{category}:{question_id}:{index:05d}",
            dataset="pope",
            split=split,
            task_type="binary_vqa",
            image=portable_image_path(dataset_dir, row.get("image"), project_root),
            image_id=image_id,
            question=clean_text(row.get("question")),
            answers=(answer,) if answer else (),
            category=category,
            has_label=bool(answer),
            metadata={
                "source_id": question_id,
                **remaining_metadata(
                    row,
                    excluded={"question_id", "image", "image_source", "question", "answer", "category"},
                    source_index=index,
                ),
            },
        )
