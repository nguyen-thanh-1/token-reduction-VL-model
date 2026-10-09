"""MMBench adapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from token_reduction_vl.data.schema import CanonicalSample, Choice

from .common import clean_text, portable_image_path, remaining_metadata, rows


def iter_mmb_samples(
    dataset_dir: Path, *, split: str, project_root: Path
) -> Iterator[CanonicalSample]:
    for index, row in enumerate(rows(dataset_dir)):
        source_id = clean_text(row.get("index")) or str(index)
        choices = tuple(
            Choice(label=label, text=text)
            for label in ("A", "B", "C", "D", "E")
            if (text := clean_text(row.get(label)))
        )
        raw_answer = clean_text(row.get("answer")).upper()
        labels = {choice.label for choice in choices}
        answer_label = raw_answer if raw_answer in labels else None
        answers = tuple(
            choice.text for choice in choices if choice.label == answer_label
        )
        category = clean_text(row.get("category")) or None
        metadata = remaining_metadata(
            row,
            excluded={
                "index",
                "question",
                "hint",
                "image",
                "answer",
                "category",
                "A",
                "B",
                "C",
                "D",
                "E",
                "split",
            },
            source_index=index,
        )
        yield CanonicalSample(
            sample_id=f"mmbench:{split}:{source_id}",
            dataset="mmbench",
            split=split,
            task_type="multiple_choice",
            image=portable_image_path(dataset_dir, row.get("image"), project_root),
            image_id=source_id,
            question=clean_text(row.get("question")),
            hint=clean_text(row.get("hint")) or None,
            choices=choices,
            answers=answers,
            answer_label=answer_label,
            category=category,
            has_label=answer_label is not None,
            metadata={"source_id": source_id, **metadata},
        )
