"""Download Hugging Face datasets and export images as ordinary files.

Each requested configuration is written as::

    <output>/<dataset>/<configuration>/
    ├── images/          # present when the dataset contains images
    ├── records.jsonl    # one metadata row per dataset row
    └── manifest.json    # completion marker and provenance

Legacy directories created with ``Dataset.save_to_disk`` are reused as the
input for export, which avoids downloading their data again.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from datasets import (
    Dataset,
    Image as DatasetImage,
    get_dataset_split_names,
    load_dataset,
    load_from_disk,
)
from PIL import Image as PILImage


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
RECORDS_FILENAME = "records.jsonl"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class DatasetRequest:
    """One Hugging Face dataset configuration to download."""

    repository: str
    config: str
    split: str
    output_subdir: Path


GQA_BALANCED_REQUESTS: tuple[DatasetRequest, ...] = tuple(
    DatasetRequest(
        repository="lmms-lab-encoder/GQA",
        config=config,
        split=split,
        output_subdir=Path("GQA") / config,
    )
    for config, split in (
        ("test_balanced_images", "test"),
        ("test_balanced_instructions", "test"),
        ("testdev_balanced_images", "testdev"),
        ("testdev_balanced_instructions", "testdev"),
        ("train_balanced_images", "train"),
        ("train_balanced_instructions", "train"),
        ("val_balanced_images", "val"),
        ("val_balanced_instructions", "val"),
    )
)

MMB_REQUEST = DatasetRequest(
    repository="HuggingFaceM4/MMBench",
    config="default",
    split="test",
    output_subdir=Path("MMB") / "test",
)

MME_REQUEST = DatasetRequest(
    repository="lmms-lab-encoder/MME",
    config="default",
    split="test",
    output_subdir=Path("MME") / "test",
)


def _resolve_split(request: DatasetRequest, token: str | None) -> str:
    """Resolve the requested split and report the actual available names."""

    split_names = get_dataset_split_names(
        request.repository,
        config_name=request.config,
        token=token,
    )
    if request.split in split_names:
        return request.split
    if len(split_names) == 1:
        return split_names[0]
    available = ", ".join(split_names)
    raise ValueError(
        f"Split {request.split!r} was not found for "
        f"{request.repository!r}/{request.config!r}. Available splits: {available}"
    )


def _safe_stem(value: Any, index: int) -> str:
    text = str(value).strip() if value is not None else ""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text[:100] or f"row_{index:08d}"


def _record_id(row: Mapping[str, Any], index: int) -> Any:
    for field in ("id", "index", "question_id", "imageId", "image_id"):
        value = row.get(field)
        if value is not None and value != "":
            return value
    return index


def _image_bytes(value: Any) -> bytes:
    """Return encoded image bytes from a datasets Image value or base64 text."""

    if isinstance(value, Mapping):
        raw = value.get("bytes")
        if raw:
            return bytes(raw)
        path = value.get("path")
        if path:
            return Path(path).read_bytes()

    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)

    if isinstance(value, PILImage.Image):
        output = BytesIO()
        image_format = value.format or "PNG"
        value.save(output, format=image_format)
        return output.getvalue()

    if isinstance(value, str):
        encoded = value.strip()
        if encoded.startswith("data:image/") and "," in encoded:
            encoded = encoded.split(",", 1)[1]
        if len(encoded) < 1024:
            try:
                candidate = Path(encoded)
                if candidate.is_file():
                    return candidate.read_bytes()
            except OSError:
                pass
        try:
            return base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError("Image text is neither a readable path nor valid base64.") from exc

    raise TypeError(f"Unsupported image value: {type(value).__name__}")


def _image_suffix(raw: bytes) -> str:
    with PILImage.open(BytesIO(raw)) as image:
        image_format = (image.format or "PNG").upper()
        image.verify()
    return {
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "GIF": ".gif",
        "BMP": ".bmp",
        "TIFF": ".tiff",
    }.get(image_format, f".{image_format.lower()}")


def _raw_image_dataset(dataset: Dataset) -> Dataset:
    feature = dataset.features.get("image")
    if isinstance(feature, DatasetImage):
        return dataset.cast_column("image", DatasetImage(decode=False))
    return dataset


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def export_dataset_files(
    dataset: Dataset,
    output_dir: Path,
    *,
    request: DatasetRequest,
    split: str,
) -> dict[str, Any]:
    """Export one dataset to JSONL metadata plus ordinary image files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / RECORDS_FILENAME
    temporary_records = output_dir / f".{RECORDS_FILENAME}.tmp"
    images_dir = output_dir / "images"
    source = _raw_image_dataset(dataset)
    has_image = "image" in source.column_names
    if has_image:
        images_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    with temporary_records.open("w", encoding="utf-8", newline="\n") as handle:
        for index, source_row in enumerate(source):
            row = dict(source_row)
            image_value = row.get("image") if has_image else None
            if image_value is not None:
                raw = _image_bytes(image_value)
                suffix = _image_suffix(raw)
                stem = _safe_stem(_record_id(row, index), index)
                image_path = images_dir / f"{index:08d}_{stem}{suffix}"
                image_path.write_bytes(raw)
                row["image"] = image_path.relative_to(output_dir).as_posix()
                image_count += 1

            handle.write(
                json.dumps(row, ensure_ascii=False, default=_json_default) + "\n"
            )
            if (index + 1) % 500 == 0:
                print(f"  exported {index + 1:,}/{len(source):,} rows")

    temporary_records.replace(records_path)
    manifest = {
        "repository": request.repository,
        "config": request.config,
        "split": split,
        "rows": len(source),
        "images": image_count,
        "records": RECORDS_FILENAME,
        "image_directory": "images" if has_image else None,
        "storage_format": "files-v1",
    }
    temporary_manifest = output_dir / f".{MANIFEST_FILENAME}.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(output_dir / MANIFEST_FILENAME)
    return manifest


def _load_legacy_dataset(output_dir: Path) -> Dataset | None:
    if not (output_dir / "dataset_info.json").is_file():
        return None
    saved = load_from_disk(str(output_dir))
    if not isinstance(saved, Dataset):
        raise TypeError(
            f"Expected a Dataset in legacy directory {output_dir}, "
            f"found {type(saved).__name__}."
        )
    return saved


def _download_request(
    request: DatasetRequest,
    output_root: Path,
    cache_dir: Path | None,
    token: str | None,
    force: bool,
    existing_only: bool,
) -> None:
    output_dir = output_root / request.output_subdir
    manifest_path = output_dir / MANIFEST_FILENAME

    if manifest_path.exists() and not force:
        print(f"[skip] {request.repository}/{request.config} -> {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_legacy_dataset(output_dir)
    split = request.split
    if dataset is not None:
        print(f"[reuse] converting existing Arrow dataset -> {output_dir}")
    else:
        if existing_only:
            print(f"[skip] no existing Arrow dataset at {output_dir}")
            return
        split = _resolve_split(request, token)
        print(
            f"[download] {request.repository} "
            f"(config={request.config}, split={split}) -> {output_dir}"
        )
        dataset = load_dataset(
            request.repository,
            name=request.config,
            split=split,
            cache_dir=str(cache_dir) if cache_dir else None,
            token=token,
        )

    manifest = export_dataset_files(
        dataset,
        output_dir,
        request=request,
        split=split,
    )
    print(
        f"[done] {manifest['rows']:,} rows and {manifest['images']:,} images "
        f"saved to {output_dir}"
    )


def _requests_for(dataset_name: str) -> Sequence[DatasetRequest]:
    if dataset_name == "gqa":
        return GQA_BALANCED_REQUESTS
    if dataset_name == "mmb":
        return (MMB_REQUEST,)
    if dataset_name == "mme":
        return (MME_REQUEST,)
    return (*GQA_BALANCED_REQUESTS, MMB_REQUEST, MME_REQUEST)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the project datasets and export image files plus JSONL metadata."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("gqa", "mmb", "mme", "all"),
        default="all",
        help="Dataset group to download (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face access token. HF_TOKEN is used when omitted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate file exports that already contain manifest.json.",
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Convert existing Arrow directories without downloading missing data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for request in _requests_for(args.dataset):
        _download_request(
            request=request,
            output_root=args.output_dir,
            cache_dir=args.cache_dir,
            token=args.token,
            force=args.force,
            existing_only=args.existing_only,
        )
