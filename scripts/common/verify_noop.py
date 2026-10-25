"""Check that the pruning wrapper with r = 100% reproduces the baseline (00-common section 4.2 step 7).

Compares run predictions with outputs/baseline/<source>.jsonl character by character.

Example:
    uv run python scripts/common/run_pruned.py --method-config none --source mme_test --level L7 --limit 100
    uv run python scripts/common/verify_noop.py outputs/none/runs/<run_id>/mme_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {row["sample_id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_file", type=Path)
    parser.add_argument("--baseline", type=Path, default=None)
    args = parser.parse_args()

    run = read_jsonl(args.run_file)
    baseline_path = args.baseline or PROJECT_ROOT / "outputs" / "baseline" / args.run_file.name
    baseline = read_jsonl(baseline_path)
    mismatches = [
        (sample_id, row["prediction"], baseline[sample_id]["prediction"])
        for sample_id, row in run.items()
        if sample_id in baseline and row["prediction"] != baseline[sample_id]["prediction"]
    ]
    token_mismatch = sum(
        1 for sample_id, row in run.items() if sample_id in baseline and row["input_tokens"] != baseline[sample_id]["input_tokens"]
    )
    compared = sum(1 for sample_id in run if sample_id in baseline)
    print(f"compared={compared} prediction_mismatches={len(mismatches)} input_token_mismatches={token_mismatch}")
    for sample_id, got, expected in mismatches[:10]:
        print(f"  {sample_id}: got {got!r}, baseline {expected!r}")
    sys.exit(1 if mismatches or token_mismatch else 0)


if __name__ == "__main__":
    main()
