"""POPE and ScienceQA-IMG (ADR 0006): adapters, prompts, metrics, target selection, image de-duplication."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from datasets import Dataset
from PIL import Image

from token_reduction_vl.data.adapters import iter_pope_samples, iter_sqa_samples
from token_reduction_vl.data.download import POPE_REQUEST, SQA_IMG_REQUEST, _requests_for, export_dataset_files
from token_reduction_vl.data.messages import build_question_prompt
from token_reduction_vl.data.preprocessing import selected_targets
from token_reduction_vl.evaluation.metrics import pope_counts, score_prediction_records


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _image(directory: Path, name: str) -> str:
    path = directory / "images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return f"images/{name}"


def _png() -> bytes:
    out = BytesIO()
    Image.new("RGB", (4, 4), "red").save(out, format="PNG")
    return out.getvalue()


def test_pope_adapter_keeps_category_and_shared_image(tmp_path: Path) -> None:
    raw = tmp_path / "POPE" / "test"
    image = _image(raw, "COCO_1.png")
    _write_jsonl(raw / "records.jsonl", [
        {"id": "0", "question_id": "1", "question": "Is there a dog in the image?", "answer": "yes",
         "image_source": "COCO_1", "category": "adversarial", "image": image},
        {"id": "1", "question_id": "1", "question": "Is there a cat in the image?", "answer": "No",
         "image_source": "COCO_1", "category": "random", "image": image},
    ])
    samples = list(iter_pope_samples(raw, split="test", project_root=tmp_path))
    assert [s.category for s in samples] == ["adversarial", "random"]
    assert samples[0].sample_id != samples[1].sample_id
    assert {s.image_id for s in samples} == {"COCO_1"}
    assert samples[1].answers == ("no",)
    assert samples[0].task_type == "binary_vqa"
    assert build_question_prompt(samples[0]).endswith("Answer yes or no only.")


def test_pope_adapter_rejects_unknown_category(tmp_path: Path) -> None:
    raw = tmp_path / "POPE" / "test"
    _write_jsonl(raw / "records.jsonl", [{"question_id": "1", "question": "q", "answer": "yes",
                                          "image_source": "a", "category": "other", "image": _image(raw, "a.png")}])
    with pytest.raises(ValueError, match="unknown category"):
        list(iter_pope_samples(raw, split="test", project_root=tmp_path))


def test_sqa_adapter_labels_choices_and_drops_explanations(tmp_path: Path) -> None:
    raw = tmp_path / "ScienceQA" / "img_test"
    _write_jsonl(raw / "records.jsonl", [
        {"question": "Which is a mammal?", "choices": ["frog", "whale", "trout"], "answer": 1,
         "hint": "Read the passage.", "subject": "natural science", "lecture": "Whales are mammals.",
         "solution": "The answer is whale.", "grade": "grade4", "image": _image(raw, "0.png")},
        {"question": "Pick one", "choices": ["a", "b"], "answer": 0, "hint": "", "subject": "language science",
         "image": _image(raw, "1.png")},
    ])
    first, second = iter_sqa_samples(raw, split="img_test", project_root=tmp_path)
    assert [c.label for c in first.choices] == ["A", "B", "C"]
    assert first.answer_label == "B" and first.answers == ("whale",)
    assert first.hint == "Read the passage." and second.hint is None
    assert "lecture" not in first.metadata and "solution" not in first.metadata
    assert first.metadata["grade"] == "grade4"
    prompt = build_question_prompt(first)
    assert prompt.startswith("Context: Read the passage.")
    assert "B. whale" in prompt and "whale." not in prompt.split("Options:")[0]
    first.validate()


def test_pope_metrics_f1_and_unparseable_answers() -> None:
    def row(pred: str, gold: str, cat: str) -> dict:
        return {"prediction": pred, "references": [gold], "metadata": {"dataset": "pope", "category": cat}}

    rows = [row("Yes", "yes", "random"), row("no", "yes", "random"), row("yes", "no", "random"),
            row("No.", "no", "random"), row("maybe", "no", "popular"), row("yes", "yes", "popular")]
    random = pope_counts(rows[:4])
    assert random["accuracy"] == 0.5
    assert random["precision"] == 0.5 and random["recall"] == 0.5 and random["f1"] == 0.5
    assert random["yes_ratio"] == 0.5
    popular = pope_counts(rows[4:])
    assert popular["accuracy"] == 0.5  # "maybe" is wrong ...
    assert popular["precision"] == 1.0  # ... and counted as "no", so precision is not hurt or helped
    result = score_prediction_records(rows)
    assert result["dataset"] == "pope"
    assert result["pope_mean_f1"] == pytest.approx((0.5 + 1.0) / 2)
    assert set(result["categories"]) == {"random", "popular"}


def test_scienceqa_metric_uses_option_letters() -> None:
    result = score_prediction_records([
        {"prediction": "B", "metadata": {"dataset": "scienceqa", "answer_label": "B",
                                         "choices": [{"label": "A", "text": "frog"}, {"label": "B", "text": "whale"}]}},
        {"prediction": "frog", "metadata": {"dataset": "scienceqa", "answer_label": "B",
                                            "choices": [{"label": "A", "text": "frog"}, {"label": "B", "text": "whale"}]}},
    ])
    assert result["accuracy"] == 0.5


def test_all_keeps_its_original_meaning() -> None:
    assert {t.dataset_group for t in selected_targets("all")} == {"gqa", "mmb", "mme"}
    assert [t.key for t in selected_targets("pope")] == ["pope_test"]
    assert [t.key for t in selected_targets("sqa")] == ["sqa_img_test"]
    assert POPE_REQUEST not in _requests_for("all") and SQA_IMG_REQUEST not in _requests_for("all")
    assert POPE_REQUEST.revision and SQA_IMG_REQUEST.revision


def test_export_stores_a_shared_image_once(tmp_path: Path) -> None:
    png = _png()
    dataset = Dataset.from_dict({
        "question_id": ["1", "2", "3"],
        "image_source": ["COCO_1", "COCO_1", "COCO_2"],
        "image": [png, png, png],
    })
    manifest = export_dataset_files(dataset, tmp_path, request=POPE_REQUEST, split="test")
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["images"] == 2 and manifest["revision"] == POPE_REQUEST.revision
    assert rows[0]["image"] == rows[1]["image"] != rows[2]["image"]
    assert len(list((tmp_path / "images").iterdir())) == 2
