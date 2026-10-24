"""MME paired yes/no adapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from token_reduction_vl.data.schema import CanonicalSample

from .common import clean_text, portable_image_path, remaining_metadata, rows


def iter_mme_samples(
    dataset_dir: Path, *, split: str, project_root: Path
) -> Iterator[CanonicalSample]:
    for index, row in enumerate(rows(dataset_dir)):
        question_id = clean_text(row.get("question_id"))
        if not question_id:
            raise ValueError(f"MME row {index} has no question_id.")
        answer = clean_text(row.get("answer"))
        category = clean_text(row.get("category")) or None
        metadata = remaining_metadata(
            row,
            excluded={"question_id", "image", "question", "answer", "category"},
            source_index=index,
        )
        yield CanonicalSample(
            sample_id=f"mme:{split}:{question_id}:{index:08d}",
            dataset="mme",
            split=split,
            task_type="binary_vqa",
            image=portable_image_path(dataset_dir, row.get("image"), project_root),
            image_id=question_id,
            question=clean_text(row.get("question")),
            answers=(answer,) if answer else (),
            category=category,
            has_label=bool(answer),
            metadata={"source_id": question_id, **metadata},
        )
