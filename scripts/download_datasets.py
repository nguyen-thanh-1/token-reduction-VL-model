"""Download the evaluation datasets used by this project.

The script saves Hugging Face datasets with ``save_to_disk`` so they can be
loaded later with ``datasets.load_from_disk`` without downloading them again.
It intentionally downloads only the balanced GQA configurations requested for
this project.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from datasets import Dataset, get_dataset_split_names, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class DatasetRequest:
    """One Hugging Face dataset configuration to download."""

    repository: str
    config: str
    split: str
    output_subdir: Path


# These names match the balanced configurations shown on the GQA dataset page.
# Keep the list explicit so *_all_* configurations cannot be downloaded by
# accident.
GQA_BALANCED_REQUESTS: tuple[DatasetRequest, ...] = tuple(
    DatasetRequest(
        repository="lmms-lab-encoder/GQA",
        config=config,
        split="train",
        output_subdir=Path("GQA") / config,
    )
    for config in (
        "test_balanced_images",
        "test_balanced_instructions",
        "testdev_balanced_images",
        "testdev_balanced_instructions",
        "train_balanced_images",
        "train_balanced_instructions",
        "val_balanced_image",
        "val_balanced_instructions",
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
    """Resolve a requested split and give a useful error if it is unavailable."""

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


def _download_request(
    request: DatasetRequest,
    output_root: Path,
    cache_dir: Path | None,
    token: str | None,
    force: bool,
) -> None:
    output_dir = output_root / request.output_subdir
    dataset_info = output_dir / "dataset_info.json"

    if dataset_info.exists() and not force:
        print(f"[skip] {request.repository}/{request.config} -> {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    split = _resolve_split(request, token)
    print(
        f"[download] {request.repository} "
        f"(config={request.config}, split={split}) -> {output_dir}"
    )

    dataset: Dataset = load_dataset(
        request.repository,
        name=request.config,
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
        token=token,
    )
    dataset.save_to_disk(str(output_dir))
    print(f"[done] {len(dataset):,} rows saved to {output_dir}")


def _requests_for(dataset_name: str) -> Sequence[DatasetRequest]:
    if dataset_name == "gqa":
        return GQA_BALANCED_REQUESTS
    if dataset_name == "mmb":
        return (MMB_REQUEST,)
    if dataset_name == "mme":
        return (MME_REQUEST,)
    return (*GQA_BALANCED_REQUESTS, MMB_REQUEST, MME_REQUEST)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Re-download entries that already contain dataset_info.json.",
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
        )


if __name__ == "__main__":
    main()
