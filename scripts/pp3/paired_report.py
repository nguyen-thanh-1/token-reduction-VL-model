"""Paired comparison of diagnostic runs on the same ids (ADR 0002).

For every run: accuracy, gained/lost answers against the full-token run, the
difference with its 95% interval, the one-sided lower bound used for the
1-point non-inferiority margin, and an exact McNemar p. ``--pairs a:b`` adds
head-to-head tests; ``--types`` adds accuracy by GQA structural type.

Example:
    uv run python scripts/pp3/paired_report.py --runs 3a_L3 3a_L4 random_L3 random_L4 \\
        --pairs 3a_L3:random_L3 3a_L4:random_L4 --types
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from token_reduction_vl.reporting.pruning_report import read_jsonl, score_prediction_record  # noqa: E402


def p_exact(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def paired(a: dict, b: dict, ids: list[str]) -> tuple[int, int, float, float, float, float, float]:
    """``a`` vs ``b``: gained, lost, diff, 95% CI low/high, one-sided 95% lower bound (points), p."""

    gained = sum(a[i] and not b[i] for i in ids)
    lost = sum(b[i] and not a[i] for i in ids)
    n = len(ids)
    d = (gained - lost) / n
    se = math.sqrt(max((gained + lost) / n - d * d, 0) / n)
    return gained, lost, 100 * d, 100 * (d - 1.96 * se), 100 * (d + 1.96 * se), 100 * (d - 1.645 * se), p_exact(gained, lost)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default="tune2k")
    parser.add_argument("--source", default="gqa_val_balanced")
    parser.add_argument("--dataset", default="gqa", help="Run-directory suffix: gqa, mmb or mme.")
    parser.add_argument("--runs", nargs="+", required=True, help="Configuration names, e.g. 3a_L3 random_L3.")
    parser.add_argument("--pairs", nargs="*", default=[], help="Head-to-head tests as a:b.")
    parser.add_argument("--types", action="store_true", help="Accuracy by GQA structural type.")
    args = parser.parse_args()

    def load(name: str) -> dict:
        method = "none" if name == "full" else "pp3_scorer"
        path = PROJECT_ROOT / "outputs" / method / "runs" / f"{args.prefix}_{name}_{args.dataset}" / f"{args.source}.jsonl"
        return {row["sample_id"]: row for row in read_jsonl(path)}

    names = list(dict.fromkeys(["full", *args.runs, *(n for pair in args.pairs for n in pair.split(":"))]))
    rows = {name: load(name) for name in names}
    ids = [i for i in rows["full"] if all(i in r for r in rows.values())]
    ok = {name: {i: bool(score_prediction_record(r[i])) for i in ids} for name, r in rows.items()}
    print(f"n = {len(ids)}  full-token accuracy = {sum(ok['full'].values()) / len(ids):.4f}\n")

    print(f"{'run':<22}{'acc':>7}{'K/N':>7}{'gain':>6}{'lost':>6}{'diff':>7}{'95% CI':>17}{'LB1s':>7}{'p':>7}")
    for name in args.runs:
        k = sum(rows[name][i]["K"] for i in ids) / sum(rows[name][i]["N"] for i in ids)
        g, l, d, lo, hi, lb, p = paired(ok[name], ok["full"], ids)
        print(f"{name:<22}{sum(ok[name].values()) / len(ids):7.4f}{k:7.3f}{g:6d}{l:6d}{d:+7.2f}   "
              f"[{lo:+5.2f},{hi:+5.2f}]{lb:+7.2f}{p:7.3f}")

    if args.pairs:
        print("\nHead-to-head (same ids):")
    for pair in args.pairs:
        a, b = pair.split(":")
        g, l, d, lo, hi, _, p = paired(ok[a], ok[b], ids)
        print(f"  {a} vs {b}: {a} wins {g}, loses {l}  diff {d:+.2f}  CI [{lo:+.2f},{hi:+.2f}]  p={p:.3f}")

    if args.types:
        groups = defaultdict(list)
        for i in ids:
            groups[(rows["full"][i]["metadata"].get("types") or {}).get("structural")].append(i)
        cols = ["full", *args.runs]
        print(f"\n{'type':<10}" + "".join(f"{c[:20]:>21}" for c in cols) + "      n")
        for t, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"{str(t):<10}" + "".join(f"{sum(ok[c][i] for i in members) / len(members):21.3f}" for c in cols)
                  + f"  {len(members):5d}")


if __name__ == "__main__":
    main()
