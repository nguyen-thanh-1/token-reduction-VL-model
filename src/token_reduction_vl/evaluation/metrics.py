"""Benchmark-specific scoring for saved baseline prediction records."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .runner import normalize_answer


def _metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("metadata", {})
    return value if isinstance(value, Mapping) else {}


def _record_field(record: Mapping[str, Any], name: str, default: Any = None) -> Any:
    value = record.get(name)
    return value if value not in (None, "") else _metadata(record).get(name, default)


def _references(record: Mapping[str, Any]) -> tuple[str, ...]:
    value = record.get("references", ())
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _gqa_correct(record: Mapping[str, Any]) -> bool | None:
    references = _references(record)
    if not references:
        return None
    prediction = normalize_answer(str(record.get("prediction", "")))
    return any(prediction == normalize_answer(answer) for answer in references)


def parse_choice_label(prediction: str, choices: Iterable[Mapping[str, Any]] = ()) -> str | None:
    """Extract an A-E label, with exact option-text matching as a fallback."""

    text = prediction.strip()
    direct = re.fullmatch(r"(?i)\s*(?:answer\s*(?:is|:)?\s*)?\(?([A-E])\)?[.)]?\s*", text)
    if direct:
        return direct.group(1).upper()
    leading = re.match(r"(?i)^\s*\(?([A-E])\)?[.)]\s+", text)
    if leading:
        return leading.group(1).upper()
    normalized = normalize_answer(text)
    for choice in choices:
        if normalized == normalize_answer(str(choice.get("text", ""))):
            return str(choice.get("label", "")).upper() or None
    return None


def parse_yes_no(prediction: str) -> str | None:
    match = re.search(r"\b(yes|no)\b", prediction, flags=re.IGNORECASE)
    return match.group(1).casefold() if match else None


def _mmb_correct(record: Mapping[str, Any]) -> bool | None:
    metadata = _metadata(record)
    answer_label = metadata.get("answer_label")
    if answer_label in (None, ""):
        return None
    choices = metadata.get("choices", ())
    if not isinstance(choices, list):
        choices = ()
    predicted = parse_choice_label(str(record.get("prediction", "")), choices)
    return predicted == str(answer_label).upper()


def _mme_correct(record: Mapping[str, Any]) -> bool | None:
    references = _references(record)
    if not references:
        return None
    predicted = parse_yes_no(str(record.get("prediction", "")))
    expected = parse_yes_no(references[0])
    return predicted is not None and expected is not None and predicted == expected


def _basic_result(dataset: str, values: list[bool | None]) -> dict[str, Any]:
    labelled = [value for value in values if value is not None]
    correct = sum(labelled)
    return {
        "dataset": dataset,
        "rows": len(values),
        "labelled_rows": len(labelled),
        "unlabelled_rows": len(values) - len(labelled),
        "correct": correct,
        "accuracy": correct / len(labelled) if labelled else None,
    }


def _score_mme(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    values = [_mme_correct(record) for record in records]
    result = _basic_result("mme", values)
    categories: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for record, correct in zip(records, values, strict=True):
        if correct is None:
            continue
        metadata = _metadata(record)
        category = str(_record_field(record, "category", "uncategorized"))
        image_id = str(_record_field(record, "image_id", record.get("sample_id", "")))
        categories[category].append((image_id, correct))

    category_metrics: dict[str, Any] = {}
    total_score = 0.0
    for category, category_rows in sorted(categories.items()):
        accuracy = 100.0 * sum(correct for _, correct in category_rows) / len(category_rows)
        pairs: dict[str, list[bool]] = defaultdict(list)
        for image_id, correct in category_rows:
            pairs[image_id].append(correct)
        complete_pairs = [pair for pair in pairs.values() if len(pair) == 2]
        accuracy_plus = (
            100.0 * sum(all(pair) for pair in complete_pairs) / len(complete_pairs)
            if complete_pairs
            else 0.0
        )
        score = accuracy + accuracy_plus
        total_score += score
        category_metrics[category] = {
            "questions": len(category_rows),
            "pairs": len(complete_pairs),
            "accuracy_percent": accuracy,
            "accuracy_plus_percent": accuracy_plus,
            "score": score,
        }
    result["mme_total_score"] = total_score
    result["categories"] = category_metrics
    return result


def score_prediction_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Score one prediction file using its canonical dataset metadata."""

    rows = list(records)
    if not rows:
        raise ValueError("Prediction file is empty.")
    datasets = {str(_record_field(row, "dataset", "")) for row in rows}
    if len(datasets) != 1 or "" in datasets:
        raise ValueError(f"Expected exactly one dataset in predictions, found {sorted(datasets)}")
    dataset = datasets.pop()
    if dataset == "gqa":
        return _basic_result(dataset, [_gqa_correct(record) for record in rows])
    if dataset == "mmbench":
        return _basic_result(dataset, [_mmb_correct(record) for record in rows])
    if dataset == "mme":
        return _score_mme(rows)
    raise ValueError(f"Unsupported benchmark dataset: {dataset!r}")
