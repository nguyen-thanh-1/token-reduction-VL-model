"""GQA balanced-split adapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from token_reduction_vl.data.schema import CanonicalSample

from .common import clean_text, portable_image_path, remaining_metadata, rows


def iter_gqa_samples(
    instructions_dir: Path,
    images_dir: Path,
    *,
    split: str,
    project_root: Path,
) -> Iterator[CanonicalSample]:
    """Join balanced GQA instructions to their image records."""

    images: dict[str, str] = {}
    for index, row in enumerate(rows(images_dir)):
        image_id = clean_text(row.get("id"))
        if not image_id:
            raise ValueError(f"GQA image row {index} has no id.")
        if image_id in images:
            raise ValueError(f"Duplicate GQA image id: {image_id!r}")
        images[image_id] = portable_image_path(
            images_dir, row.get("image"), project_root
        )

    if not images:
        raise ValueError(f"No GQA image records found in {images_dir}.")

    for index, row in enumerate(rows(instructions_dir)):
        source_id = clean_text(row.get("id"))
        image_id = clean_text(row.get("imageId"))
        question = clean_text(row.get("question"))
        if not source_id or not image_id or not question:
            raise ValueError(f"Incomplete GQA instruction row {index} in {instructions_dir}.")
        if image_id not in images:
            raise KeyError(f"GQA instruction {source_id!r} references missing image {image_id!r}.")

        answer = clean_text(row.get("answer"))
        metadata = remaining_metadata(
            row,
            excluded={"id", "imageId", "question", "answer"},
            source_index=index,
        )
        yield CanonicalSample(
            sample_id=f"gqa:{split}:{source_id}",
            dataset="gqa",
            split=split,
            task_type="open_vqa",
            image=images[image_id],
            image_id=image_id,
            question=question,
            answers=(answer,) if answer else (),
            has_label=bool(answer),
            metadata={"source_id": source_id, **metadata},
        )
