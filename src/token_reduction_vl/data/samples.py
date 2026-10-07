"""Convert saved Hugging Face datasets into a common VQA sample format."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image


DEFAULT_IMAGE_FIELDS = ("image", "image_path", "image_file", "img")
DEFAULT_QUESTION_FIELDS = ("question", "instruction", "prompt", "query", "text")
DEFAULT_ANSWER_FIELDS = ("answer", "answers", "label", "response", "gt_answer")
DEFAULT_ID_FIELDS = ("question_id", "id", "uid", "image_id", "imageId")


@dataclass(frozen=True)
class VQASample:
    """One image-question example in the project-wide representation."""

    sample_id: str
    image: Any
    question: str
    answers: tuple[str, ...]
    metadata: dict[str, Any]


def _first_value(row: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        value = row.get(field)
        if value is not None and value != "":
            return value
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(_as_text(item) for item in value)
    return str(value).strip()


def _as_answers(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        nested = value.get("answers", value.get("answer", value.get("text")))
        return _as_answers(nested)
    if isinstance(value, (list, tuple, set)):
        return tuple(answer for answer in (_as_text(item) for item in value) if answer)
    text = _as_text(value)
    return (text,) if text else ()


def _open_image(value: Any) -> Any:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, Mapping):
        raw_bytes = value.get("bytes")
        if raw_bytes:
            return Image.open(BytesIO(raw_bytes)).convert("RGB")
        value = value.get("path")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    if value is None:
        raise ValueError("The dataset row does not contain an image value.")
    return value


def _build_question(row: Mapping[str, Any], question_fields: Sequence[str]) -> str:
    question = _as_text(_first_value(row, question_fields))
    if not question:
        raise ValueError("The dataset row does not contain a question or instruction.")

    context: list[str] = []
    hint = _as_text(row.get("hint"))
    if hint:
        context.append(f"Hint: {hint}")
    choices = row.get("choices")
    if choices:
        context.append(f"Options: {_as_text(choices)}")
    return "\n".join([question, *context])


def row_to_sample(
    row: Mapping[str, Any],
    index: int,
    *,
    image_fields: Sequence[str] = DEFAULT_IMAGE_FIELDS,
    question_fields: Sequence[str] = DEFAULT_QUESTION_FIELDS,
    answer_fields: Sequence[str] = DEFAULT_ANSWER_FIELDS,
    id_fields: Sequence[str] = DEFAULT_ID_FIELDS,
) -> VQASample:
    """Convert one raw dataset row into a :class:`VQASample`."""

    sample_id = _as_text(_first_value(row, id_fields)) or str(index)
    image = _open_image(_first_value(row, image_fields))
    question = _build_question(row, question_fields)
    answers = _as_answers(_first_value(row, answer_fields))
    return VQASample(
        sample_id=sample_id,
        image=image,
        question=question,
        answers=answers,
        metadata={"source_index": index},
    )


def load_saved_dataset(path: Path) -> Dataset:
    """Load a Dataset or single-split DatasetDict saved with ``save_to_disk``."""

    saved = load_from_disk(str(path))
    if isinstance(saved, Dataset):
        return saved
    if isinstance(saved, DatasetDict):
        if len(saved) == 1:
            return next(iter(saved.values()))
        if "train" in saved:
            return saved["train"]
        raise ValueError(f"Expected one split in {path}, found {list(saved)}")
    raise TypeError(f"Unsupported saved dataset type at {path}: {type(saved).__name__}")


def iter_single_samples(
    path: Path,
    *,
    image_fields: Sequence[str] = DEFAULT_IMAGE_FIELDS,
    question_fields: Sequence[str] = DEFAULT_QUESTION_FIELDS,
    answer_fields: Sequence[str] = DEFAULT_ANSWER_FIELDS,
    id_fields: Sequence[str] = DEFAULT_ID_FIELDS,
) -> Iterable[VQASample]:
    dataset = load_saved_dataset(path)
    for index, row in enumerate(dataset):
        yield row_to_sample(
            row,
            index,
            image_fields=image_fields,
            question_fields=question_fields,
            answer_fields=answer_fields,
            id_fields=id_fields,
        )


def _join_key(row: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    value = _first_value(row, fields)
    text = _as_text(value)
    return text or None


def iter_paired_samples(
    instructions_path: Path,
    images_path: Path,
    *,
    join_key_fields: Sequence[str],
    image_fields: Sequence[str] = DEFAULT_IMAGE_FIELDS,
    question_fields: Sequence[str] = DEFAULT_QUESTION_FIELDS,
    answer_fields: Sequence[str] = DEFAULT_ANSWER_FIELDS,
    id_fields: Sequence[str] = DEFAULT_ID_FIELDS,
) -> Iterable[VQASample]:
    """Join instruction rows with image rows, currently used by GQA."""

    instructions = load_saved_dataset(instructions_path)
    images = load_saved_dataset(images_path)
    image_by_key: dict[str, Mapping[str, Any]] = {}
    for row in images:
        key = _join_key(row, join_key_fields)
        if key is not None:
            image_by_key[key] = row

    if not image_by_key:
        raise ValueError(
            f"No image join keys found in {images_path}. Tried fields: {join_key_fields}"
        )

    for index, instruction_row in enumerate(instructions):
        key = _join_key(instruction_row, join_key_fields)
        if key is None or key not in image_by_key:
            raise KeyError(
                f"Could not match instruction row {index} to an image using key {key!r}."
            )
        row = {**image_by_key[key], **instruction_row}
        yield row_to_sample(
            row,
            index,
            image_fields=image_fields,
            question_fields=question_fields,
            answer_fields=answer_fields,
            id_fields=id_fields,
        )
