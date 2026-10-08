from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

from datasets import Dataset, Features, Image as DatasetImage, Value
from PIL import Image

from token_reduction_vl.data.download import (
    GQA_BALANCED_REQUESTS,
    DatasetRequest,
    _download_request,
    _load_legacy_dataset,
    export_dataset_files,
)
from token_reduction_vl.data.samples import iter_single_samples, row_to_sample


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (3, 2), color=(12, 34, 56)).save(output, format="PNG")
    return output.getvalue()


def test_gqa_requests_are_balanced_and_use_plural_val_images() -> None:
    configs = {request.config for request in GQA_BALANCED_REQUESTS}

    assert "val_balanced_images" in configs
    assert "val_balanced_image" not in configs
    assert all("_all_" not in config for config in configs)


def test_export_writes_real_image_and_relative_jsonl_path(tmp_path) -> None:
    image_bytes = _png_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    dataset = Dataset.from_dict(
        {"index": [7], "question": ["What is shown?"], "image": [encoded]}
    )
    request = DatasetRequest("example/dataset", "default", "test", tmp_path)

    manifest = export_dataset_files(
        dataset,
        tmp_path,
        request=request,
        split="test",
    )

    record = json.loads((tmp_path / "records.jsonl").read_text(encoding="utf-8"))
    image_path = tmp_path / record["image"]
    assert image_path.suffix == ".png"
    assert image_path.read_bytes() == image_bytes
    assert manifest["rows"] == 1
    assert manifest["images"] == 1


def test_legacy_arrow_image_dataset_is_reused_for_export(tmp_path) -> None:
    legacy_dir = tmp_path / "legacy"
    output_dir = tmp_path / "output"
    dataset = Dataset.from_dict(
        {"id": ["img-1"], "image": [{"bytes": _png_bytes(), "path": None}]},
        features=Features({"id": Value("string"), "image": DatasetImage()}),
    )
    dataset.save_to_disk(str(legacy_dir))

    loaded = _load_legacy_dataset(legacy_dir)
    assert loaded is not None
    export_dataset_files(
        loaded,
        output_dir,
        request=DatasetRequest("example/gqa", "balanced_images", "test", output_dir),
        split="test",
    )

    record = json.loads((output_dir / "records.jsonl").read_text(encoding="utf-8"))
    assert (output_dir / record["image"]).read_bytes() == _png_bytes()


def test_existing_only_never_downloads_missing_dataset(tmp_path, monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("network-backed dataset lookup should not be called")

    monkeypatch.setattr(
        "token_reduction_vl.data.download.get_dataset_split_names",
        fail_if_called,
    )
    monkeypatch.setattr("token_reduction_vl.data.download.load_dataset", fail_if_called)

    _download_request(
        DatasetRequest("example/missing", "default", "test", Path("missing")),
        output_root=tmp_path,
        cache_dir=None,
        token=None,
        force=False,
        existing_only=True,
    )

    assert not (tmp_path / "missing" / "manifest.json").exists()


def test_file_export_can_be_loaded_as_vqa_sample(tmp_path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "sample.png").write_bytes(_png_bytes())
    record = {
        "question_id": "q1",
        "question": "Choose one",
        "answer": "A",
        "A": "first",
        "B": "second",
        "image": "images/sample.png",
    }
    (tmp_path / "records.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    sample = next(iter(iter_single_samples(tmp_path)))

    assert sample.sample_id == "q1"
    assert sample.image.size == (3, 2)
    assert "A. first" in sample.question
    assert "B. second" in sample.question
    assert sample.answers == ("A",)


def test_legacy_mmb_base64_image_is_decoded() -> None:
    sample = row_to_sample(
        {
            "index": 9,
            "question": "What color?",
            "answer": "A",
            "image": base64.b64encode(_png_bytes()).decode("ascii"),
        },
        0,
    )

    assert sample.sample_id == "9"
    assert sample.image.size == (3, 2)
