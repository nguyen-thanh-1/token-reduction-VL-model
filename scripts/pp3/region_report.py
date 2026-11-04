"""Shape of the kept-token sets of diagnostic runs (goal 2), next to their accuracy.

Columns: accuracy, K/N, connected groups (and per 10 kept tokens), share of
kept tokens with a kept neighbour, coverage of 3x3-token cells, and the mean
overlap (Jaccard) of keep sets between different questions on the same image.
A selection that follows the question has low overlap; random selection gives
the floor for each K/N. Rows are split into yes/no-answer and other questions.

Example:
    uv run python scripts/pp3/region_report.py --runs random_L3 3a_L3 3d_attn50k_L3 random_L4 3a_L4 3d_attn50k_L4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from token_reduction_vl.reporting.pruning_report import read_jsonl, score_prediction_record  # noqa: E402
from token_reduction_vl.reporting.region_metrics import summarize  # noqa: E402

COLUMNS = ("acc", "K/N", "groups", "grp/10", "nbr%", "cover%", "q-overlap")


def line(name: str, rows: list[dict]) -> str:
    s = summarize(rows)
    if not s["rows"]:
        return f"{name:<24}(no keep_indices)"
    acc = sum(bool(score_prediction_record(row)) for row in rows) / len(rows)
    overlap = "-" if s["question_overlap"] is None else f"{s['question_overlap']:.3f}"
    text = (f"{name:<24}{acc:8.4f}{s['keep_ratio']:8.3f}{s['components']:8.1f}{s['components_per_10']:8.2f}"
            f"{100 * s['neighbor_fraction']:8.1f}{100 * s['coverage']:8.1f}{overlap:>10}  n={s['rows']}")
    # Focus-context runs: the same metrics on the focus tokens alone.
    focus_rows = [
        {**row, "keep_indices": row["pruner_metadata"]["focus_indices"], "K": len(row["pruner_metadata"]["focus_indices"])}
        for row in rows if (row.get("pruner_metadata") or {}).get("focus_indices")
    ]
    if focus_rows:
        f = summarize(focus_rows)
        share = sum(row["pruner_metadata"]["focus_share"] for row in focus_rows) / len(focus_rows)
        text += (f"\n{'  focus part':<24}{'':8}{f['keep_ratio']:8.3f}{f['components']:8.1f}{f['components_per_10']:8.2f}"
                 f"{100 * f['neighbor_fraction']:8.1f}{100 * f['coverage']:8.1f}{f['question_overlap']:>10.3f}"
                 f"  share={share:.2f}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default="tune2k")
    parser.add_argument("--source", default="gqa_val_balanced")
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()

    header = f"{'run':<24}" + "".join(f"{c:>8}" for c in COLUMNS[:-1]) + f"{COLUMNS[-1]:>10}"
    groups: dict[str, list[str]] = {"all": [], "yes/no answer": [], "other answer": []}
    for name in args.runs:
        path = PROJECT_ROOT / "outputs" / "pp3_scorer" / "runs" / f"{args.prefix}_{name}_gqa" / f"{args.source}.jsonl"
        rows = read_jsonl(path)
        yes_no = [row for row in rows if str(row["references"][0]).strip().lower() in ("yes", "no")]
        other = [row for row in rows if row not in yes_no]
        groups["all"].append(line(name, rows))
        groups["yes/no answer"].append(line(name, yes_no))
        groups["other answer"].append(line(name, other))
    for title, lines in groups.items():
        print(f"\n== {title}\n{header}")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
