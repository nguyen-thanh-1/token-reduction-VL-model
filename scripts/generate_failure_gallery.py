"""Generate a self-contained HTML gallery of incorrect benchmark answers."""

from __future__ import annotations

import argparse
from pathlib import Path

from token_reduction_vl.config import load_yaml
from token_reduction_vl.reporting import build_failure_gallery


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "report_baseline.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional HTML output path; defaults to the configured report directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output = args.output
    if output is not None and not output.is_absolute():
        output = PROJECT_ROOT / output
    result = build_failure_gallery(
        config,
        project_root=PROJECT_ROOT,
        output_path=output,
    )
    print(
        f"Generated {result['cases']} failure cases with "
        f"{result['unique_images']} unique images: {result['path']}"
    )


if __name__ == "__main__":
    main()
