"""Runtime prompt/message construction for canonical samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .samples import VQASample, _open_image
from .schema import CanonicalSample


def build_question_prompt(sample: CanonicalSample) -> str:
    """Build a task-aware prompt without adding model-specific tokens."""

    sections: list[str] = []
    if sample.hint:
        sections.append(f"Context: {sample.hint}")
    sections.append(sample.question)
    if sample.choices:
        sections.append(
            "Options:\n"
            + "\n".join(
                f"{choice.label}. {choice.text}" for choice in sample.choices
            )
        )
        sections.append("Answer with the option letter only.")
    elif sample.task_type == "binary_vqa":
        sections.append("Answer yes or no only.")
    else:
        sections.append("Answer with a short phrase.")
    return "\n\n".join(sections)


def answer_target(sample: CanonicalSample) -> str | None:
    """Return the preferred supervised target for one sample."""

    if sample.answer_label:
        return sample.answer_label
    return sample.answers[0] if sample.answers else None


def to_qwen_messages(
    sample: CanonicalSample,
    *,
    image: Any | None = None,
    include_answer: bool = False,
) -> list[dict[str, Any]]:
    """Create Qwen-compatible messages at runtime.

    ``include_answer=True`` is intended for supervised training examples and
    is rejected when the source split has no label.
    """

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image if image is not None else sample.image},
                {"type": "text", "text": build_question_prompt(sample)},
            ],
        }
    ]
    if include_answer:
        target = answer_target(sample)
        if target is None:
            raise ValueError(f"Sample {sample.sample_id!r} does not have a training label.")
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target}]}
        )
    return messages


def canonical_to_vqa(sample: CanonicalSample, *, project_root: Path) -> VQASample:
    """Load the referenced image and adapt a canonical record to the runner."""

    image = _open_image(sample.image, base_path=project_root)
    metadata = {
        **sample.metadata,
        "schema_version": sample.schema_version,
        "dataset": sample.dataset,
        "split": sample.split,
        "task_type": sample.task_type,
        "image_id": sample.image_id,
        "answer_label": sample.answer_label,
        "choices": [choice.to_dict() for choice in sample.choices],
        "category": sample.category,
        "has_label": sample.has_label,
    }
    return VQASample(
        sample_id=sample.sample_id,
        image=image,
        question=build_question_prompt(sample),
        answers=sample.answers,
        metadata=metadata,
    )
