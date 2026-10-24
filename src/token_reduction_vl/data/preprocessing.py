"""Create versioned canonical JSONL files from downloaded raw datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .adapters import iter_gqa_samples, iter_mmb_samples, iter_mme_samples
from .schema import SCHEMA_VERSION, CanonicalSample


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "v1"


@dataclass(frozen=True)
class TargetSpec:
    key: str
    dataset_group: str
    output: Path
    split: str
    source_dirs: tuple[Path, ...]


TARGETS: tuple[TargetSpec, ...] = (
    *(
        TargetSpec(
            key=f"gqa_{split}",
            dataset_group="gqa",
            output=Path("gqa") / f"{split}.jsonl",
            split=split,
            source_dirs=(
                Path("GQA") / f"{split}_balanced_instructions",
                Path("GQA") / f"{split}_balanced_images",
            ),
        )
        for split in ("train", "val", "testdev", "test")
    ),
    TargetSpec(
        key="mmb_validation",
        dataset_group="mmb",
        output=Path("mmbench") / "validation.jsonl",
        split="validation",
        source_dirs=(Path("MMB") / "validation",),
    ),
    TargetSpec(
        key="mmb_test",
        dataset_group="mmb",
        output=Path("mmbench") / "test.jsonl",
        split="test",
        source_dirs=(Path("MMB") / "test",),
    ),
    TargetSpec(
        key="mme_test",
        dataset_group="mme",
        output=Path("mme") / "test.jsonl",
        split="test",
        source_dirs=(Path("MME") / "test",),
    ),
)


def selected_targets(dataset_group: str, splits: tuple[str, ...] = ()) -> tuple[TargetSpec, ...]:
    targets = tuple(
        target
        for target in TARGETS
        if dataset_group == "all" or target.dataset_group == dataset_group
    )
    if splits:
        requested = set(splits)
        targets = tuple(target for target in targets if target.split in requested)
        missing = requested - {target.split for target in targets}
        if missing:
            raise ValueError(
                f"Splits not available for {dataset_group!r}: {', '.join(sorted(missing))}"
            )
    return targets


def _iter_target(
    target: TargetSpec, *, raw_root: Path, project_root: Path
) -> Iterator[CanonicalSample]:
    paths = tuple(raw_root / source for source in target.source_dirs)
    missing = [path for path in paths if not (path / "records.jsonl").is_file()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Raw records are missing for {target.key}: {formatted}. "
            "Run scripts/download_datasets.py first."
        )
    if target.dataset_group == "gqa":
        return iter_gqa_samples(
            paths[0], paths[1], split=target.split, project_root=project_root
        )
    if target.dataset_group == "mmb":
        return iter_mmb_samples(paths[0], split=target.split, project_root=project_root)
    return iter_mme_samples(paths[0], split=target.split, project_root=project_root)


def _resolve_image(sample: CanonicalSample, project_root: Path) -> Path:
    path = Path(sample.image)
    return path if path.is_absolute() else project_root / path


def _source_provenance(target: TargetSpec, raw_root: Path, project_root: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for relative_dir in target.source_dirs:
        source_dir = raw_root / relative_dir
        records_path = source_dir / "records.jsonl"
        manifest_path = source_dir / "manifest.json"
        try:
            portable_records = records_path.relative_to(project_root).as_posix()
        except ValueError:
            portable_records = records_path.resolve().as_posix()
        item: dict[str, Any] = {
            "records": portable_records,
            "size_bytes": records_path.stat().st_size,
        }
        if manifest_path.is_file():
            item["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources.append(item)
    return sources


def write_processed_target(
    target: TargetSpec,
    samples: Iterable[CanonicalSample],
    output_path: Path,
    *,
    project_root: Path,
    raw_root: Path,
    overwrite: bool = False,
    limit: int | None = None,
    check_images: bool = True,
) -> dict[str, Any]:
    """Validate and atomically write one processed split."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Processed output already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    sample_ids: set[str] = set()
    checked_images: set[str] = set()
    image_ids: set[str] = set()
    task_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    labelled = 0
    count = 0

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in samples:
                if limit is not None and count >= limit:
                    break
                sample.validate()
                if sample.sample_id in sample_ids:
                    raise ValueError(f"Duplicate sample_id: {sample.sample_id!r}")
                sample_ids.add(sample.sample_id)
                image_ids.add(sample.image_id)
                if check_images and sample.image not in checked_images:
                    image_path = _resolve_image(sample, project_root)
                    if not image_path.is_file():
                        raise FileNotFoundError(
                            f"Image for {sample.sample_id!r} does not exist: {image_path}"
                        )
                    checked_images.add(sample.image)

                handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
                task_counts[sample.task_type] += 1
                if sample.category:
                    category_counts[sample.category] += 1
                labelled += int(sample.has_label)
                count += 1
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    resolved_output = output_path.resolve()
    try:
        portable_output = resolved_output.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        portable_output = resolved_output.as_posix()
    return {
        "dataset": target.dataset_group,
        "split": target.split,
        "path": portable_output,
        "rows": count,
        "labelled_rows": labelled,
        "unlabelled_rows": count - labelled,
        "unique_image_ids": len(image_ids),
        "unique_image_paths_checked": len(checked_images) if check_images else None,
        "task_counts": dict(sorted(task_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "limited": limit is not None,
        "sources": _source_provenance(target, raw_root, project_root),
    }


def preprocess_targets(
    targets: Iterable[TargetSpec],
    *,
    raw_root: Path = DEFAULT_RAW_DIR,
    output_root: Path = DEFAULT_OUTPUT_DIR,
    project_root: Path = PROJECT_ROOT,
    overwrite: bool = False,
    limit: int | None = None,
    check_images: bool = True,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for target in targets:
        print(f"[preprocess] {target.key} -> {output_root / target.output}")
        artifacts[target.key] = write_processed_target(
            target,
            _iter_target(target, raw_root=raw_root, project_root=project_root),
            output_root / target.output,
            project_root=project_root,
            raw_root=raw_root,
            overwrite=overwrite,
            limit=limit,
            check_images=check_images,
        )
        print(f"[done] {artifacts[target.key]['rows']:,} rows")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    existing_artifacts: dict[str, Any] = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("schema_version") == SCHEMA_VERSION:
            existing_artifacts = dict(existing.get("artifacts", {}))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts": {**existing_artifacts, **artifacts},
    }
    temporary_manifest = output_root / ".manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("gqa", "mmb", "mme", "all"), default="all"
    )
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="Optional split filter; repeat for multiple splits.",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Rows per split for a smoke run.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip checking that every referenced raw image exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    targets = selected_targets(args.dataset, tuple(args.split))
    preprocess_targets(
        targets,
        raw_root=args.raw_dir,
        output_root=args.output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        check_images=not args.skip_image_check,
    )
