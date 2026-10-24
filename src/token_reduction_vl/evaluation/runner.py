"""Simple JSONL runner for reproducible full-token baseline results."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from token_reduction_vl.data import VQASample


class PredictiveModel(Protocol):
    def predict(self, image: Any, question: str) -> Any: ...


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def _exact_match(prediction: str, answers: tuple[str, ...]) -> bool | None:
    if not answers:
        return None
    normalized_prediction = normalize_answer(prediction)
    return any(normalized_prediction == normalize_answer(answer) for answer in answers)


def run_baseline(
    samples: Iterable[VQASample],
    model: PredictiveModel,
    output_path: Path,
    *,
    limit: int | None = None,
) -> int:
    """Run inference and write one JSON object per sample."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in tqdm(samples, total=limit, desc="baseline"):
            if limit is not None and count >= limit:
                break
            result = model.predict(sample.image, sample.question)
            task_type = sample.metadata.get("task_type")
            record = {
                "sample_id": sample.sample_id,
                "dataset": sample.metadata.get("dataset"),
                "split": sample.metadata.get("split"),
                "task_type": task_type,
                "image_id": sample.metadata.get("image_id"),
                "question": sample.question,
                "references": list(sample.answers),
                "prediction": result.text,
                "exact_match": (
                    _exact_match(result.text, sample.answers)
                    if task_type in (None, "open_vqa")
                    else None
                ),
                "latency_seconds": result.latency_seconds,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "peak_vram_mb": result.peak_vram_mb,
                "pruning_method": "none",
                "metadata": sample.metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
