"""Convert saved Hugging Face datasets into a common VQA sample format."""

from __future__ import annotations

import base64
import json
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
DEFAULT_ID_FIELDS = ("question_id", "id", "index", "uid", "image_id", "imageId")


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


def _open_image(value: Any, *, base_path: Path | None = None) -> Any:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, Mapping):
        raw_bytes = value.get("bytes")
        if raw_bytes:
            return Image.open(BytesIO(raw_bytes)).convert("RGB")
        value = value.get("path")
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        candidate = Path(text)
        if not candidate.is_absolute() and base_path is not None:
            candidate = base_path / candidate
        try:
            if candidate.is_file():
                return Image.open(candidate).convert("RGB")
        except OSError:
            pass
        if text.startswith("data:image/") and "," in text:
            text = text.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(text, validate=True)
        except ValueError as exc:
            raise ValueError(
                f"Image value is neither a readable path nor valid base64: {value!r}"
            ) from exc
        return Image.open(BytesIO(raw_bytes)).convert("RGB")
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
    if not choices:
        labelled_choices = [
            f"{label}. {_as_text(row.get(label))}"
            for label in ("A", "B", "C", "D", "E")
            if _as_text(row.get(label))
        ]
        choices = "\n".join(labelled_choices)
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
    image_base_path: Path | None = None,
) -> VQASample:
    """Convert one raw dataset row into a :class:`VQASample`."""

    sample_id = _as_text(_first_value(row, id_fields)) or str(index)
    image = _open_image(_first_value(row, image_fields), base_path=image_base_path)
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
    """Load a legacy Dataset or single-split DatasetDict."""

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


def iter_saved_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    """Stream exported JSONL rows or iterate a legacy Arrow dataset."""

    records_path = path / "records.jsonl"
    if records_path.is_file():
        with records_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Expected an object at {records_path}:{line_number}."
                    )
                yield value
        return

    yield from load_saved_dataset(path)


def iter_single_samples(
    path: Path,
    *,
    image_fields: Sequence[str] = DEFAULT_IMAGE_FIELDS,
    question_fields: Sequence[str] = DEFAULT_QUESTION_FIELDS,
    answer_fields: Sequence[str] = DEFAULT_ANSWER_FIELDS,
    id_fields: Sequence[str] = DEFAULT_ID_FIELDS,
) -> Iterable[VQASample]:
    for index, row in enumerate(iter_saved_rows(path)):
        yield row_to_sample(
            row,
            index,
            image_fields=image_fields,
            question_fields=question_fields,
            answer_fields=answer_fields,
            id_fields=id_fields,
            image_base_path=path,
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

    image_by_key: dict[str, Mapping[str, Any]] = {}
    for row in iter_saved_rows(images_path):
        key = _join_key(row, join_key_fields)
        if key is not None:
            image_by_key[key] = row

    if not image_by_key:
        raise ValueError(
            f"No image join keys found in {images_path}. Tried fields: {join_key_fields}"
        )

    for index, instruction_row in enumerate(iter_saved_rows(instructions_path)):
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
            image_base_path=images_path,
        )
