from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from token_reduction_vl.evaluation.metrics import score_prediction_record
from token_reduction_vl.reporting.failure_gallery import build_failure_gallery


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_benchmark_specific_single_record_scoring() -> None:
    assert score_prediction_record(
        {"dataset": "gqa", "prediction": "Woman", "references": ["woman"]}
    ) is True
    assert score_prediction_record(
        {
            "dataset": "mmbench",
            "prediction": "B",
            "metadata": {
                "answer_label": "A",
                "choices": [
                    {"label": "A", "text": "cat"},
                    {"label": "B", "text": "dog"},
                ],
            },
        }
    ) is False
    assert score_prediction_record(
        {"dataset": "mme", "prediction": "No.", "references": ["Yes"]}
    ) is False


def test_failure_gallery_embeds_images_and_only_includes_wrong_rows(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "data" / "sample.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (48, 32), color=(35, 90, 160)).save(image_path)
    canonical_path = tmp_path / "canonical.jsonl"
    _write_jsonl(
        canonical_path,
        [
            {
                "sample_id": "wrong",
                "dataset": "gqa",
                "split": "testdev",
                "image": "data/sample.png",
                "image_id": "image-1",
                "question": "What color is the object?",
                "hint": None,
                "choices": [],
                "answers": ["blue"],
                "answer_label": None,
                "category": "color",
                "metadata": {},
            },
            {
                "sample_id": "correct",
                "dataset": "gqa",
                "split": "testdev",
                "image": "data/sample.png",
                "image_id": "image-1",
                "question": "Is it blue?",
                "hint": None,
                "choices": [],
                "answers": ["yes"],
                "answer_label": None,
                "category": "verify",
                "metadata": {},
            },
        ],
    )
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions_path,
        [
            {
                "sample_id": "wrong",
                "dataset": "gqa",
                "prediction": "red",
                "references": ["blue"],
                "input_tokens": 50,
                "output_tokens": 2,
                "latency_seconds": 0.1,
            },
            {
                "sample_id": "correct",
                "dataset": "gqa",
                "prediction": "yes",
                "references": ["yes"],
                "input_tokens": 50,
                "output_tokens": 2,
                "latency_seconds": 0.1,
            },
        ],
    )
    output = tmp_path / "report" / "failure_cases.html"
    config = {
        "report": {
            "title": "Fixture baseline",
            "model_id": "fixture/model",
            "output_dir": "report",
        },
        "predictions": {
            "gqa": {"label": "GQA fixture", "path": "predictions.jsonl"}
        },
        "failure_gallery": {
            "canonical_sources": {"gqa": "canonical.jsonl"},
            "max_cases_per_dataset": 10,
            "image_max_width": 128,
            "jpeg_quality": 75,
        },
    }

    result = build_failure_gallery(config, project_root=tmp_path, output_path=output)

    html = output.read_text(encoding="utf-8")
    assert result["cases"] == 1
    assert result["unique_images"] == 1
    assert "data:image/jpeg;base64," in html
    assert "What color is the object?" in html
    assert '"prediction":"red"' in html
    assert '"reference":"blue"' in html
    assert '"sample_id":"correct"' not in html
