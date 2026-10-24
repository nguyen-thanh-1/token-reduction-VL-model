from __future__ import annotations

import json
from pathlib import Path

import pytest

from token_reduction_vl.data.adapters import (
    iter_gqa_samples,
    iter_mmb_samples,
    iter_mme_samples,
)
from token_reduction_vl.data.messages import answer_target, build_question_prompt, to_qwen_messages
from token_reduction_vl.data.preprocessing import TargetSpec, preprocess_targets, selected_targets
from token_reduction_vl.data.schema import SCHEMA_VERSION, CanonicalSample, Choice
from token_reduction_vl.evaluation.metrics import (
    parse_choice_label,
    score_prediction_records,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _image(directory: Path, name: str = "sample.png") -> str:
    path = directory / "images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-decoded-by-preprocessing")
    return f"images/{name}"


def test_schema_round_trip_and_training_messages() -> None:
    sample = CanonicalSample(
        sample_id="mmbench:validation:7",
        dataset="mmbench",
        split="validation",
        task_type="multiple_choice",
        image="data/raw/MMB/validation/images/7.png",
        image_id="7",
        question="Which answer?",
        hint="Inspect the diagram.",
        choices=(Choice("A", "first"), Choice("B", "second")),
        answers=("second",),
        answer_label="B",
        category="reasoning",
        has_label=True,
    )

    restored = CanonicalSample.from_dict(sample.to_dict())
    messages = to_qwen_messages(restored, include_answer=True)

    assert restored.schema_version == SCHEMA_VERSION
    assert "B. second" in build_question_prompt(restored)
    assert answer_target(restored) == "B"
    assert messages[-1]["content"][0]["text"] == "B"


def test_unlabelled_sample_cannot_be_used_as_training_target() -> None:
    sample = CanonicalSample(
        sample_id="gqa:test:q1",
        dataset="gqa",
        split="test",
        task_type="open_vqa",
        image="image.png",
        image_id="i1",
        question="What is shown?",
    )

    with pytest.raises(ValueError, match="does not have a training label"):
        to_qwen_messages(sample, include_answer=True)


def test_gqa_adapter_joins_images_and_preserves_annotations(tmp_path: Path) -> None:
    images = tmp_path / "data" / "raw" / "GQA" / "val_balanced_images"
    instructions = tmp_path / "data" / "raw" / "GQA" / "val_balanced_instructions"
    image_path = _image(images)
    _write_jsonl(images / "records.jsonl", [{"id": "img-1", "image": image_path}])
    _write_jsonl(
        instructions / "records.jsonl",
        [
            {
                "id": "q-1",
                "imageId": "img-1",
                "question": "Is it blue?",
                "answer": "yes",
                "fullAnswer": "Yes, it is blue.",
                "types": {"semantic": "attr"},
            }
        ],
    )

    sample = next(
        iter_gqa_samples(
            instructions, images, split="val", project_root=tmp_path
        )
    )

    assert sample.sample_id == "gqa:val:q-1"
    assert sample.image == "data/raw/GQA/val_balanced_images/images/sample.png"
    assert sample.answers == ("yes",)
    assert sample.metadata["fullAnswer"] == "Yes, it is blue."


def test_mmb_and_mme_adapters_handle_labels_and_duplicate_source_ids(tmp_path: Path) -> None:
    mmb = tmp_path / "data" / "raw" / "MMB" / "validation"
    mme = tmp_path / "data" / "raw" / "MME" / "test"
    _write_jsonl(
        mmb / "records.jsonl",
        [
            {
                "index": 3,
                "question": "Pick one",
                "A": "cat",
                "B": "dog",
                "answer": "B",
                "image": _image(mmb),
                "category": "attribute",
            }
        ],
    )
    shared_image = _image(mme)
    _write_jsonl(
        mme / "records.jsonl",
        [
            {
                "question_id": "existence/1.png",
                "question": "Is there a cat?",
                "answer": "Yes",
                "category": "existence",
                "image": shared_image,
            },
            {
                "question_id": "existence/1.png",
                "question": "Is there a dog?",
                "answer": "No",
                "category": "existence",
                "image": shared_image,
            },
        ],
    )

    mmb_sample = next(iter_mmb_samples(mmb, split="validation", project_root=tmp_path))
    mme_samples = list(iter_mme_samples(mme, split="test", project_root=tmp_path))

    assert mmb_sample.answer_label == "B"
    assert mmb_sample.answers == ("dog",)
    assert mme_samples[0].sample_id != mme_samples[1].sample_id
    assert mme_samples[0].image_id == mme_samples[1].image_id == "existence/1.png"


def test_preprocess_writes_versioned_jsonl_and_manifest(tmp_path: Path) -> None:
    images = tmp_path / "data" / "raw" / "GQA" / "val_balanced_images"
    instructions = tmp_path / "data" / "raw" / "GQA" / "val_balanced_instructions"
    _write_jsonl(
        images / "records.jsonl",
        [{"id": "img-1", "image": _image(images)}],
    )
    _write_jsonl(
        instructions / "records.jsonl",
        [{"id": "q-1", "imageId": "img-1", "question": "What?", "answer": "x"}],
    )
    for directory in (images, instructions):
        (directory / "manifest.json").write_text(
            json.dumps({"rows": 1, "storage_format": "files-v1"}), encoding="utf-8"
        )
    target = TargetSpec(
        key="gqa_val",
        dataset_group="gqa",
        output=Path("gqa/val.jsonl"),
        split="val",
        source_dirs=(
            Path("GQA/val_balanced_instructions"),
            Path("GQA/val_balanced_images"),
        ),
    )
    output_root = tmp_path / "data" / "processed" / "v1"

    manifest = preprocess_targets(
        (target,),
        raw_root=tmp_path / "data" / "raw",
        output_root=output_root,
        project_root=tmp_path,
    )

    record = json.loads((output_root / "gqa" / "val.jsonl").read_text(encoding="utf-8"))
    assert record["sample_id"] == "gqa:val:q-1"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["artifacts"]["gqa_val"]["rows"] == 1


def test_split_selection_rejects_unknown_split() -> None:
    assert {target.split for target in selected_targets("mme")} == {"test"}
    with pytest.raises(ValueError, match="not available"):
        selected_targets("mme", ("validation",))


def test_benchmark_specific_metrics() -> None:
    assert parse_choice_label("Answer: B") == "B"
    assert parse_choice_label("second", [{"label": "B", "text": "second"}]) == "B"

    mmb = score_prediction_records(
        [
            {
                "prediction": "B",
                "references": ["dog"],
                "metadata": {
                    "dataset": "mmbench",
                    "answer_label": "B",
                    "choices": [{"label": "B", "text": "dog"}],
                },
            }
        ]
    )
    mme = score_prediction_records(
        [
            {
                "sample_id": "1a",
                "prediction": "Yes",
                "references": ["Yes"],
                "metadata": {"dataset": "mme", "category": "existence", "image_id": "1"},
            },
            {
                "sample_id": "1b",
                "prediction": "No",
                "references": ["No"],
                "metadata": {"dataset": "mme", "category": "existence", "image_id": "1"},
            },
        ]
    )

    assert mmb["accuracy"] == 1.0
    assert mme["mme_total_score"] == 200.0
