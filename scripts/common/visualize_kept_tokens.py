"""Draw which visual tokens a pruned run kept, on the original image.

Each visual token covers a 32x32 px cell of the processor-resized image; dropped
cells are darkened, kept cells are outlined in green. The caption shows the
question, the prediction, the baseline prediction and the reference, and K/N.
Writes PNGs and an index.html gallery into --output-dir (default: <run>/kept_tokens/).

Example:
    uv run python scripts/common/visualize_kept_tokens.py outputs/pp3_scorer/runs/<run_id>/gqa_testdev_balanced.jsonl --num 24 --filter changed
"""

from __future__ import annotations

import argparse
import html
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from PIL import Image  # noqa: E402

from token_reduction_vl.data.canonical import iter_canonical_records  # noqa: E402
from token_reduction_vl.evaluation.metrics import score_prediction_record  # noqa: E402
from token_reduction_vl.reporting.pruning_report import read_jsonl, render_kept_tokens  # noqa: E402

FILTERS = ("all", "correct", "wrong", "changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_file", type=Path)
    parser.add_argument("--num", type=int, default=24)
    parser.add_argument("--filter", choices=FILTERS, default="all", help="changed = prediction differs from the baseline.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.run_file) if row.get("keep_indices")]
    if not rows:
        raise SystemExit("The run has no keep_indices (was it run with --no-save-keep?).")
    baseline_path = PROJECT_ROOT / "outputs" / "baseline" / args.run_file.name
    baseline = {row["sample_id"]: row for row in read_jsonl(baseline_path)} if baseline_path.exists() else {}
    if args.filter == "correct":
        rows = [row for row in rows if score_prediction_record(row) is True]
    elif args.filter == "wrong":
        rows = [row for row in rows if score_prediction_record(row) is False]
    elif args.filter == "changed":
        rows = [row for row in rows if row["sample_id"] in baseline and row["prediction"] != baseline[row["sample_id"]]["prediction"]]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.num]
    if not rows:
        raise SystemExit(f"No rows match --filter {args.filter}.")

    dataset, split = rows[0]["dataset"], rows[0]["split"]
    canonical = PROJECT_ROOT / "data" / "processed" / "v1" / dataset / f"{split}.jsonl"
    wanted = {row["sample_id"] for row in rows}
    images = {record.sample_id: record.image for record in iter_canonical_records(canonical) if record.sample_id in wanted}

    output_dir = args.output_dir or args.run_file.parent / "kept_tokens" / f"{args.run_file.stem}_{args.filter}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        image = Image.open(PROJECT_ROOT / images[row["sample_id"]])
        rendered = render_kept_tokens(image, tuple(row["grid_hw"]), row["keep_indices"])
        name = row["sample_id"].replace(":", "_") + ".png"
        rendered.save(output_dir / name)
        correct = score_prediction_record(row)
        base = baseline.get(row["sample_id"], {}).get("prediction", "—")
        question = row["question"].split("\n\n")
        cards.append(
            f"<figure><img src='{name}' loading='lazy'><figcaption>"
            f"<b>{html.escape(row['sample_id'])}</b> · K={row['K']}/{row['N']} ({row['K'] / row['N']:.1%}) · input {row['input_tokens']} tokens<br>"
            f"Q: {html.escape(' / '.join(question[:-1]) or row['question'])}<br>"
            f"pred: <b>{html.escape(row['prediction'])}</b> ({'✓' if correct else '✗' if correct is False else '?'}) · "
            f"baseline: {html.escape(base)} · ref: {html.escape(', '.join(row['references']))}"
            "</figcaption></figure>"
        )
    page = (
        "<!doctype html><meta charset='utf-8'><title>Kept tokens</title>"
        "<style>body{font-family:sans-serif;margin:16px}figure{display:inline-block;width:420px;margin:8px;vertical-align:top}"
        "img{max-width:100%;border:1px solid #ccc}figcaption{font-size:12px}</style>"
        f"<h3>{html.escape(str(args.run_file))} — filter: {args.filter}</h3>"
        "<p>Green outline = kept visual token (32×32 px of the resized image); dark = dropped.</p>" + "".join(cards)
    )
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"Wrote {len(rows)} images and {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
