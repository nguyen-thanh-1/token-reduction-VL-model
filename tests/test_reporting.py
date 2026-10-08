from __future__ import annotations

import json
from pathlib import Path

import pytest

from token_reduction_vl.reporting.benchmark import (
    build_benchmark_report,
    summarize_performance,
)


def _prediction(
    sample_id: str,
    dataset: str,
    prediction: str,
    reference: str,
    *,
    latency: float,
    metadata: dict | None = None,
) -> dict:
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "prediction": prediction,
        "references": [reference],
        "latency_seconds": latency,
        "input_tokens": 100,
        "output_tokens": 2,
        "peak_vram_mb": 4096.0,
        "metadata": {"dataset": dataset, **(metadata or {})},
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _fixture_config(tmp_path: Path) -> dict:
    predictions = tmp_path / "predictions"
    _write_jsonl(
        predictions / "gqa.jsonl",
        [
            _prediction("g1", "gqa", "cat", "cat", latency=0.1),
            _prediction("g2", "gqa", "dog", "cat", latency=0.2),
        ],
    )
    _write_jsonl(
        predictions / "mmb.jsonl",
        [
            _prediction(
                "b1",
                "mmbench",
                "A",
                "cat",
                latency=0.2,
                metadata={
                    "answer_label": "A",
                    "choices": [{"label": "A", "text": "cat"}],
                },
            ),
            _prediction(
                "b2",
                "mmbench",
                "B",
                "cat",
                latency=0.3,
                metadata={
                    "answer_label": "A",
                    "choices": [{"label": "A", "text": "cat"}],
                },
            ),
        ],
    )
    _write_jsonl(
        predictions / "mme.jsonl",
        [
            _prediction(
                "e1a",
                "mme",
                "yes",
                "Yes",
                latency=0.3,
                metadata={"category": "existence", "image_id": "image-1"},
            ),
            _prediction(
                "e1b",
                "mme",
                "no",
                "No",
                latency=0.4,
                metadata={"category": "existence", "image_id": "image-1"},
            ),
        ],
    )
    return {
        "report": {
            "id": "test-report",
            "title": "Test report",
            "output_dir": "report",
            "model_id": "test/model",
            "precision": "bf16",
            "pruning_method": "none",
        },
        "predictions": {
            "gqa": {"label": "GQA", "path": "predictions/gqa.jsonl"},
            "mmbench": {"label": "MMBench", "path": "predictions/mmb.jsonl"},
            "mme": {"label": "MME", "path": "predictions/mme.jsonl"},
        },
        "visualization": {"dpi": 50, "max_scatter_points_per_dataset": 10},
    }


def test_performance_summary_detects_duplicates_and_empty_predictions() -> None:
    records = [
        _prediction("same", "gqa", "cat", "cat", latency=0.1),
        _prediction("same", "gqa", "", "cat", latency=0.3),
    ]

    summary = summarize_performance(records)

    assert summary["rows"] == 2
    assert summary["duplicate_sample_ids"] == 1
    assert summary["empty_predictions"] == 1
    assert summary["mean_latency_seconds"] == pytest.approx(0.2)


def test_report_has_separate_figures_tables_and_manifest(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    output = tmp_path / "report"

    manifest = build_benchmark_report(
        config,
        project_root=tmp_path,
        output_root=output,
    )

    assert manifest["report_id"] == "test-report"
    assert (output / "report.md").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "tables" / "benchmark_summary.csv").is_file()
    assert (output / "tables" / "mme_categories.csv").is_file()
    assert len(list((output / "figures").glob("*.png"))) == 5

    with pytest.raises(FileExistsError, match="--overwrite"):
        build_benchmark_report(config, project_root=tmp_path, output_root=output)
