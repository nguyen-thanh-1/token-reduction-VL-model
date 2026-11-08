"""Final-evaluation tables: quality per benchmark and efficiency, one row per run.

Reads runs written by run_diagnostics.py with --split all, i.e.
outputs/<method>/runs/<prefix>_<name>_<dataset>/<source>.jsonl. Benchmarks are
never averaged (CLAUDE.md): GQA and MMBench report accuracy with the paired
difference to full tokens; MME reports the perception and cognition scores
(sum of acc + acc+ per subtask; not a percentage). Efficiency is reported
separately: kept share of visual tokens, mean LLM input length, selector,
prefill and decode time per sample, and peak VRAM. ViT time is excluded
because the runs share one ViT pass.

Example:
    uv run python scripts/pp3/final_report.py --runs random_L3 divprune_L3 fastv_L3 3d_attn50k_L3 \\
        --output outputs/reports/pp3_final/tables.md
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from token_reduction_vl.evaluation.metrics import score_prediction_record, score_prediction_records  # noqa: E402
from token_reduction_vl.reporting.pruning_report import read_jsonl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_report import paired  # noqa: E402

BENCHMARKS = (("gqa", "gqa_testdev_balanced"), ("mmb", "mmb_validation"), ("mme", "mme_test"))
MME_COGNITION = frozenset({"commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"})


def run_path(prefix: str, name: str, dataset: str, source: str) -> Path:
    method = "none" if name == "full" else "pp3_scorer"
    return PROJECT_ROOT / "outputs" / method / "runs" / f"{prefix}_{name}_{dataset}" / f"{source}.jsonl"


def mme_scores(rows: list[dict]) -> tuple[float, float]:
    categories = score_prediction_records(rows)["categories"]
    perception = sum(c["score"] for name, c in categories.items() if name not in MME_COGNITION)
    cognition = sum(c["score"] for name, c in categories.items() if name in MME_COGNITION)
    return perception, cognition


def quality_table(prefix: str, runs: list[str]) -> list[str]:
    header = [*(("MME perc.", "MME cogn.") if d == "mme" else (f"{d.upper()} acc", "diff (95% CI)")
                for d, _ in BENCHMARKS)]
    header += [("POPE F1", "POPE yes %") for d, _ in BENCHMARKS if d == "pope"]
    columns = [c for pair in header for c in pair]
    lines = ["| Run | K/N | " + " | ".join(columns) + " |", "|---|---|" + "---|" * len(columns)]
    loaded = {(name, d): read_jsonl(run_path(prefix, name, d, s)) if run_path(prefix, name, d, s).is_file() else None
              for name in ["full", *runs] for d, s in BENCHMARKS}
    for name in ["full", *runs]:
        cells, ratios = [], []
        for dataset, _ in BENCHMARKS:
            rows, full = loaded[(name, dataset)], loaded[("full", dataset)]
            if rows is None:
                cells += ["-", "-"]
                continue
            ratios.append(sum(r["K"] for r in rows) / sum(r["N"] for r in rows))
            if dataset == "mme":
                perception, cognition = mme_scores(rows)
                cells += [f"{perception:.1f}", f"{cognition:.1f}"]
                continue
            ok = {r["sample_id"]: bool(score_prediction_record(r)) for r in rows}
            cells.append(f"{100 * sum(ok.values()) / len(ok):.2f}")
            if name == "full" or full is None:
                cells.append("")
                continue
            ref = {r["sample_id"]: bool(score_prediction_record(r)) for r in full}
            ids = [i for i in ok if i in ref]
            if len(ids) != len(ok) or len(ids) != len(ref):
                print(f"WARNING: {name}/{dataset}: {len(ok)} rows vs {len(ref)} full rows; "
                      f"diff uses the {len(ids)} shared ids, acc columns are not comparable", file=sys.stderr)
            _, _, d, lo, hi, _, _ = paired(ok, ref, ids)
            cells.append(f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        if any(d == "pope" for d, _ in BENCHMARKS):
            rows = loaded[(name, "pope")]
            if rows is None:
                cells += ["-", "-"]
            else:
                pope = score_prediction_records(rows)
                yes = statistics.mean(c["yes_ratio"] for c in pope["categories"].values())
                cells += [f"{100 * pope['pope_mean_f1']:.2f}", f"{100 * yes:.1f}"]
        ratio = f"{statistics.mean(ratios):.3f}" if ratios else "-"
        lines.append(f"| {name} | {ratio} | " + " | ".join(cells) + " |")
    return lines


def efficiency_table(prefix: str, runs: list[str], dataset: str, source: str) -> list[str]:
    lines = [f"| Run ({dataset}) | K/N | LLM input tokens | selector ms | prefill ms | decode ms | peak VRAM MB |",
             "|---|---|---|---|---|---|---|"]
    for name in ["full", *runs]:
        path = run_path(prefix, name, dataset, source)
        if not path.is_file():
            continue
        rows = read_jsonl(path)
        mean = lambda key: 1000 * statistics.mean(r["timing"].get(key, 0.0) for r in rows)  # noqa: E731
        vram = [r["peak_vram_mb"] for r in rows if r.get("peak_vram_mb") is not None]
        lines.append(
            f"| {name} | {sum(r['K'] for r in rows) / sum(r['N'] for r in rows):.3f} | "
            f"{statistics.mean(r['input_tokens'] for r in rows):.0f} | {mean('selector'):.1f} | "
            f"{mean('prefill'):.1f} | {mean('decode'):.1f} | {max(vram) if vram else float('nan'):.0f} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default="final")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=None, help="Also write the tables to this Markdown file.")
    parser.add_argument("--benchmark", action="append", default=None,
                        help="dataset:source to report instead of the three test sets, e.g. gqa:gqa_val_balanced "
                             "or pope:pope_test / sqa:sqa_img_test (ADR 0006).")
    args = parser.parse_args()
    global BENCHMARKS
    if args.benchmark:
        BENCHMARKS = tuple(tuple(item.split(":", 1)) for item in args.benchmark)

    text = ["## Quality (each benchmark separately)", "", *quality_table(args.prefix, args.runs), ""]
    text += ["## Efficiency (ViT excluded, shared pass)", ""]
    for dataset, source in BENCHMARKS:
        text += [*efficiency_table(args.prefix, args.runs, dataset, source), ""]
    report = "\n".join(text)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
