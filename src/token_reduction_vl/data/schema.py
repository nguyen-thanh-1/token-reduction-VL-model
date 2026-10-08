"""Versioned, model-agnostic schema for processed VQA examples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


SCHEMA_VERSION = "1.0"
TaskType = Literal["open_vqa", "multiple_choice", "binary_vqa"]
SUPPORTED_TASK_TYPES = frozenset({"open_vqa", "multiple_choice", "binary_vqa"})


@dataclass(frozen=True)
class Choice:
    """One labelled multiple-choice option."""

    label: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "text": self.text}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Choice:
        return cls(label=str(value["label"]), text=str(value["text"]))


@dataclass(frozen=True)
class CanonicalSample:
    """One portable VQA record shared by training, inference, and evaluation."""

    sample_id: str
    dataset: str
    split: str
    task_type: TaskType
    image: str
    image_id: str
    question: str
    hint: str | None = None
    choices: tuple[Choice, ...] = ()
    answers: tuple[str, ...] = ()
    answer_label: str | None = None
    category: str | None = None
    has_label: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        """Raise ``ValueError`` when a record violates schema invariants."""

        required = {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "split": self.split,
            "image": self.image,
            "image_id": self.image_id,
            "question": self.question,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"Missing required canonical fields: {', '.join(missing)}")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}."
            )
        if self.task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"Unsupported task type: {self.task_type!r}")

        labels = [choice.label for choice in self.choices]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate choice labels in sample {self.sample_id!r}.")
        if self.task_type == "multiple_choice" and len(self.choices) < 2:
            raise ValueError(
                f"Multiple-choice sample {self.sample_id!r} needs at least two choices."
            )
        if self.answer_label is not None and self.answer_label not in labels:
            raise ValueError(
                f"Answer label {self.answer_label!r} is not present in choices for "
                f"sample {self.sample_id!r}."
            )
        expected_has_label = bool(self.answers or self.answer_label)
        if self.has_label != expected_has_label:
            raise ValueError(
                f"has_label={self.has_label!r} disagrees with answers for "
                f"sample {self.sample_id!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "split": self.split,
            "task_type": self.task_type,
            "image": self.image,
            "image_id": self.image_id,
            "question": self.question,
            "hint": self.hint,
            "choices": [choice.to_dict() for choice in self.choices],
            "answers": list(self.answers),
            "answer_label": self.answer_label,
            "category": self.category,
            "has_label": self.has_label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CanonicalSample:
        choices = tuple(Choice.from_dict(item) for item in value.get("choices", ()))
        sample = cls(
            schema_version=str(value.get("schema_version", "")),
            sample_id=str(value.get("sample_id", "")),
            dataset=str(value.get("dataset", "")),
            split=str(value.get("split", "")),
            task_type=str(value.get("task_type", "")),  # type: ignore[arg-type]
            image=str(value.get("image", "")),
            image_id=str(value.get("image_id", "")),
            question=str(value.get("question", "")),
            hint=(str(value["hint"]) if value.get("hint") not in (None, "") else None),
            choices=choices,
            answers=tuple(str(answer) for answer in value.get("answers", ())),
            answer_label=(
                str(value["answer_label"])
                if value.get("answer_label") not in (None, "")
                else None
            ),
            category=(
                str(value["category"])
                if value.get("category") not in (None, "")
                else None
            ),
            has_label=bool(value.get("has_label", False)),
            metadata=dict(value.get("metadata", {})),
        )
        sample.validate()
        return sample
