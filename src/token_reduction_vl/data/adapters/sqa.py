"""ScienceQA-IMG multiple-choice adapter (ADR 0006)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from token_reduction_vl.data.schema import CanonicalSample, Choice

from .common import clean_text, portable_image_path, remaining_metadata, rows

LABELS = "ABCDE"


def iter_sqa_samples(
    dataset_dir: Path, *, split: str, project_root: Path
) -> Iterator[CanonicalSample]:
    """Choices are labelled A–E in source order; ``answer`` is a 0-based index."""

    for index, row in enumerate(rows(dataset_dir)):
        texts = [clean_text(text) for text in (row.get("choices") or [])]
        if not 2 <= len(texts) <= len(LABELS):
            raise ValueError(f"ScienceQA row {index} has {len(texts)} choices.")
        choices = tuple(Choice(label=LABELS[i], text=text) for i, text in enumerate(texts))
        raw_answer = row.get("answer")
        answer_label = (
            LABELS[int(raw_answer)]
            if raw_answer not in (None, "") and 0 <= int(raw_answer) < len(choices)
            else None
        )
        answers = tuple(choice.text for choice in choices if choice.label == answer_label)
        source_id = f"{index:05d}"
        yield CanonicalSample(
            sample_id=f"scienceqa:{split}:{source_id}",
            dataset="scienceqa",
            split=split,
            task_type="multiple_choice",
            image=portable_image_path(dataset_dir, row.get("image"), project_root),
            image_id=source_id,
            question=clean_text(row.get("question")),
            hint=clean_text(row.get("hint")) or None,
            choices=choices,
            answers=answers,
            answer_label=answer_label,
            category=clean_text(row.get("subject")) or None,
            has_label=answer_label is not None,
            metadata={
                "source_id": source_id,
                **remaining_metadata(
                    row,
                    # lecture/solution are explanations of the answer; dropped so they cannot leak into prompts.
                    excluded={"image", "question", "choices", "answer", "hint", "subject", "lecture", "solution"},
                    source_index=index,
                ),
            },
        )
