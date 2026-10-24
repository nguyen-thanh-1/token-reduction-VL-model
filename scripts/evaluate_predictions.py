"""Score a saved baseline JSONL file with benchmark-specific metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from token_reduction_vl.evaluation.metrics import score_prediction_records


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            yield value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = score_prediction_records(_read_jsonl(args.predictions))
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote metrics to {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
