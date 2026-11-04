"""Offline simulation of a per-question budget rule from finished diagnostic runs.

Generation is greedy, so a question answered with configuration X gives the
answer already stored in X's run. A rule "question form -> configuration"
(forms from ``question_form``: text only, no annotations) is scored by taking,
for each question, the row of the configuration its form maps to. Output: the
form mix against GQA structural types, then accuracy, mean K/N (sum K / sum N)
and the paired test against full tokens for every rule on the Pareto front
(fewest tokens for its accuracy), plus explicit ``--policy`` rules.

The rule is chosen on tune2k; confirm it on a second split (ADR 0002).

Example:
    uv run python scripts/pp3/simulate_gate.py
    uv run python scripts/pp3/simulate_gate.py --policy yesno=3d_attn50k_L5,choice=fcd25_L3,open=fcd25_L3
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from token_reduction_vl.pruning.pp3_scorer.question_form import FORMS, question_form  # noqa: E402
from token_reduction_vl.reporting.pruning_report import read_jsonl, score_prediction_record  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_report import paired  # noqa: E402

DEFAULT_CANDIDATES = [f"{head}_{level}" for head in ("3d_attn50k", "fcd25") for level in ("L2", "L3", "L4", "L5")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default="tune2k")
    parser.add_argument("--source", default="gqa_val_balanced")
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--policy", action="append", default=[], help="form=run,... e.g. yesno=3d_attn50k_L5,choice=fcd25_L3,open=fcd25_L3")
    parser.add_argument("--top", type=int, default=15, help="Pareto rules to print.")
    args = parser.parse_args()

    def load(name: str) -> dict:
        method = "none" if name == "full" else "pp3_scorer"
        path = PROJECT_ROOT / "outputs" / method / "runs" / f"{args.prefix}_{name}_gqa" / f"{args.source}.jsonl"
        return {row["sample_id"]: row for row in read_jsonl(path)}

    policies = [dict(item.split("=", 1) for item in text.split(",")) for text in args.policy]
    names = list(dict.fromkeys(["full", *args.candidates, *(run for p in policies for run in p.values())]))
    rows = {name: load(name) for name in names}
    ids = [i for i in rows["full"] if all(i in r for r in rows.values())]
    ok = {name: {i: bool(score_prediction_record(r[i])) for i in ids} for name, r in rows.items()}
    full = rows["full"]
    form = {i: question_form(full[i]["question"], full[i]["metadata"].get("choices")) for i in ids}
    n_tokens = sum(full[i]["N"] for i in ids)

    print(f"n = {len(ids)}  full-token accuracy = {sum(ok['full'].values()) / len(ids):.4f}")
    mix = Counter((form[i], (full[i]["metadata"].get("types") or {}).get("structural")) for i in ids)
    types = sorted({t for _, t in mix}, key=str)
    print(f"\nform \\ GQA type " + "".join(f"{str(t):>9}" for t in types) + "    total")
    for f in FORMS:
        print(f"{f:<16}" + "".join(f"{mix[(f, t)]:9d}" for t in types) + f"{sum(mix[(f, t)] for t in types):9d}")

    print(f"\nPer form: accuracy of each candidate (full first)")
    print(f"{'run':<18}" + "".join(f"{f:>9}" for f in FORMS))
    for name in ["full", *args.candidates]:
        accs = []
        for f in FORMS:
            members = [i for i in ids if form[i] == f]
            accs.append(sum(ok[name][i] for i in members) / max(1, len(members)))
        print(f"{name:<18}" + "".join(f"{a:9.3f}" for a in accs))

    def evaluate(policy: dict[str, str]) -> tuple:
        chosen = {i: ok[policy[form[i]]][i] for i in ids}
        kept = sum(rows[policy[form[i]]][i]["K"] for i in ids) / n_tokens
        return (sum(chosen.values()) / len(ids), kept, *paired(chosen, ok["full"], ids))

    header = f"{'yesno':<14}{'choice':<14}{'open':<14}{'acc':>7}{'K/N':>7}{'gain':>6}{'lost':>6}{'diff':>7}{'95% CI':>17}{'LB1s':>7}{'p':>7}"

    def show(policy: dict[str, str], result: tuple) -> str:
        acc, kept, g, l, d, lo, hi, lb, p = result
        return (f"{policy['yesno']:<14}{policy['choice']:<14}{policy['open']:<14}{acc:7.4f}{kept:7.3f}{g:6d}{l:6d}"
                f"{d:+7.2f}   [{lo:+5.2f},{hi:+5.2f}]{lb:+7.2f}{p:7.3f}")

    results = []
    for combo in itertools.product(args.candidates, repeat=len(FORMS)):
        policy = dict(zip(FORMS, combo))
        results.append((policy, evaluate(policy)))
    results.sort(key=lambda item: item[1][1])
    front, best = [], -1.0
    for policy, result in results:
        if result[0] > best:
            front.append((policy, result))
            best = result[0]
    print(f"\nPareto front over {len(results)} rules (fewest tokens for each accuracy), last {args.top}:")
    print(header)
    for policy, result in front[-args.top:]:
        print(show(policy, result))
    if policies:
        print("\nGiven rules:")
        print(header)
        for policy in policies:
            print(show(policy, evaluate(policy)))


if __name__ == "__main__":
    main()
