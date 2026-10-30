"""Summarize pruned runs: tables + plots, each run compared with the baseline on the same sample ids.

Writes summary.csv, summary.md, quality_vs_keep_<dataset>.png,
latency_breakdown_<dataset>.png and token_usage_*.png into --output-dir.

Example:
    uv run python scripts/common/summarize_runs.py --runs "outputs/pp3_scorer/runs/*" --output-dir outputs/pp3_scorer/reports/dev_1k
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from token_reduction_vl.reporting.pruning_report import (  # noqa: E402
    collect_summaries,
    plot_summaries,
    plot_token_usage,
    write_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", action="append", required=True, help="Run directory or glob; repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=PROJECT_ROOT / "outputs" / "baseline")
    parser.add_argument("--exclude", default="smoke_", help="Skip run directories whose name contains this text.")
    args = parser.parse_args()

    run_dirs = sorted(
        {Path(path) for pattern in args.runs for path in glob.glob(str(PROJECT_ROOT / pattern) if not Path(pattern).is_absolute() else pattern)}
    )
    run_dirs = [path for path in run_dirs if path.is_dir() and (not args.exclude or args.exclude not in path.name)]
    if not run_dirs:
        raise SystemExit("No run directories matched.")
    summaries = collect_summaries(run_dirs, args.baseline_dir)
    write_tables(summaries, args.output_dir)
    figures = plot_summaries(summaries, args.output_dir)
    figures += plot_token_usage((p for d in run_dirs for p in d.glob("*.jsonl") if p.name != "errors.jsonl"), args.output_dir / "token_usage")
    print(f"{len(summaries)} run files summarized -> {args.output_dir / 'summary.md'}; {len(figures)} figures")


if __name__ == "__main__":
    main()
