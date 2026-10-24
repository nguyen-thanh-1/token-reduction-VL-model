"""Generate a structured benchmark report from full-run prediction JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from token_reduction_vl.config import load_yaml
from token_reduction_vl.reporting import build_benchmark_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "report_baseline.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for report.output_dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate the same report directory when it already contains files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = args.output_dir
    if output_root is not None and not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    manifest = build_benchmark_report(
        config,
        project_root=PROJECT_ROOT,
        output_root=output_root,
        overwrite=args.overwrite,
    )
    configured_output = output_root or (
        PROJECT_ROOT / config["report"]["output_dir"]
    )
    print(
        f"Generated {len(manifest['generated_files']) + 1} report artifacts in "
        f"{configured_output}"
    )


if __name__ == "__main__":
    main()
