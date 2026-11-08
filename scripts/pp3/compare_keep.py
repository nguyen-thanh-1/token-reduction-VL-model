"""Compare kept token sets and predictions of the same configuration in two run prefixes.

Used to check that a faster implementation selects the same tokens as the runs
already reported (e.g. the CUDA-graph DPP against the ``final`` runs).

Example:
    uv run python scripts/pp3/compare_keep.py --old final --new parity --source mme_test --dataset mme --runs 3d_attn50k_L3 3d_attn50k_L4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from token_reduction_vl.reporting.pruning_report import read_jsonl  # noqa: E402
from final_report import run_path  # noqa: E402


def keep_set(row: dict) -> frozenset[int]:
    return frozenset(row.get("keep_indices") or row.get("keep") or [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", default="final")
    parser.add_argument("--new", required=True)
    parser.add_argument("--source", default="gqa_testdev_balanced")
    parser.add_argument("--dataset", default="gqa")
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    for name in args.runs:
        old = {r["sample_id"]: r for r in read_jsonl(run_path(args.old, name, args.dataset, args.source))}
        new = {r["sample_id"]: r for r in read_jsonl(run_path(args.new, name, args.dataset, args.source))}
        ids = [i for i in new if i in old]
        same_keep = sum(keep_set(old[i]) == keep_set(new[i]) for i in ids)
        same_pred = sum(old[i]["prediction"] == new[i]["prediction"] for i in ids)
        jaccard = [len(keep_set(old[i]) & keep_set(new[i])) / max(1, len(keep_set(old[i]) | keep_set(new[i]))) for i in ids]
        print(f"{name}: n={len(ids)}  identical keep sets {same_keep} ({100 * same_keep / max(1, len(ids)):.1f}%)  "
              f"mean Jaccard {sum(jaccard) / max(1, len(jaccard)):.4f}  identical predictions {same_pred}")


if __name__ == "__main__":
    main()
