"""Oracle denoising test (ADR 0002, gate before stage 3): do harmful visual tokens exist?

For each sample the signed gradient × input of the *ground-truth* answer is
computed with all tokens. The tokens with the most negative contribution
(they lower p(ground truth) to first order) are dropped without folding, and the
model answers again. A random drop of the same size is the control.

The oracle uses the reference answer, so it is an upper bound, not a method:
if even the oracle barely improves accuracy, "beat the baseline by removing
noise" is not worth a new label set.

Rows go to outputs/pp3_scorer/runs/<run-id>/<source>.jsonl (resumable) and the
summary to summary.md next to them.

Example:
    uv run python scripts/pp3/oracle_denoise.py --limit 5 --run-id smoke_oracle
    uv run python scripts/pp3/oracle_denoise.py
    uv run python scripts/pp3/oracle_denoise.py --summary-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from token_reduction_vl.config import load_yaml  # noqa: E402
from token_reduction_vl.data.splits import iter_split_samples, read_split  # noqa: E402
from token_reduction_vl.evaluation.metrics import score_prediction_record  # noqa: E402
from token_reduction_vl.evaluation.run_meta import utc_now, write_run_meta  # noqa: E402
from token_reduction_vl.models.qwen3_vl_pruned import PrunedQwen3VL, question_text_for_scoring  # noqa: E402
from token_reduction_vl.pruning.base import NoOpPruner  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.diagnostics import KeepIndicesPruner, complement, most_negative  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.dsfold_base import DSFoldParams  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.saliency import MAX_ANSWER_TOKENS, signed_gradient_x_input  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "common"))
from run_pruned import finished_rows  # noqa: E402

BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline_qwen3_vl.yaml"
NO_FOLD = DSFoldParams().with_overrides(gamma=0.0)


def correct(sample, prediction: str) -> bool:
    record = {"dataset": sample.metadata.get("dataset"), "prediction": prediction,
              "references": list(sample.answers), "metadata": sample.metadata}
    return bool(score_prediction_record(record))


def answer(runner, sample, scoring_text, pruner, encoded) -> str:
    return runner.predict(sample.image, sample.question, scoring_text, pruner, None, encoded=encoded).text


def process(runner, sample, fractions, generator) -> dict:
    scoring_text = question_text_for_scoring(sample.question, sample.metadata.get("choices") or [])
    encoded = runner.encode(sample.image, sample.question, scoring_text)
    n = encoded.num_visual
    target = runner.processor.tokenizer(sample.answers[0], add_special_tokens=False)["input_ids"][:MAX_ANSWER_TOKENS]
    signed = signed_gradient_x_input(runner, encoded, torch.tensor(target)).cpu()
    full = answer(runner, sample, scoring_text, NoOpPruner(), encoded)
    row = {
        "sample_id": sample.sample_id,
        "references": list(sample.answers),
        "structural": (sample.metadata.get("types") or {}).get("structural"),
        "N": n,
        "negative_fraction": float((signed < 0).float().mean()),
        "full": {"prediction": full, "correct": correct(sample, full)},
        "drops": {},
    }
    for fraction in fractions:
        drop = most_negative(signed, fraction)
        random_drop = torch.randperm(n, generator=generator)[: drop.numel()]
        oracle_text = answer(runner, sample, scoring_text, KeepIndicesPruner(complement(n, drop), NO_FOLD), encoded)
        random_text = answer(runner, sample, scoring_text, KeepIndicesPruner(complement(n, random_drop), NO_FOLD), encoded)
        row["drops"][f"{fraction:g}"] = {
            "dropped": int(drop.numel()),
            "oracle": {"prediction": oracle_text, "correct": correct(sample, oracle_text)},
            "random": {"prediction": random_text, "correct": correct(sample, random_text)},
        }
    return row


def p_exact(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def summarize(rows: list[dict]) -> str:
    n = len(rows)
    base = sum(r["full"]["correct"] for r in rows)
    wrong = [r for r in rows if not r["full"]["correct"]]
    lines = [
        f"# Oracle denoising test ({n} samples)",
        "",
        f"Full-token accuracy {base / n:.4f}; baseline wrong on {len(wrong)}. "
        f"Mean share of tokens with negative signed saliency: {sum(r['negative_fraction'] for r in rows) / n:.3f}.",
        "",
        "| drop | arm | mean dropped | accuracy | gained | lost | diff (pts) | 95% CI | p |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in rows[0]["drops"]:
        for arm in ("oracle", "random"):
            g = sum(r["drops"][key][arm]["correct"] and not r["full"]["correct"] for r in rows)
            lost = sum(r["full"]["correct"] and not r["drops"][key][arm]["correct"] for r in rows)
            d = (g - lost) / n
            se = math.sqrt(max((g + lost) / n - d * d, 0) / n)
            dropped = sum(r["drops"][key]["dropped"] / r["N"] for r in rows) / n
            acc = sum(r["drops"][key][arm]["correct"] for r in rows) / n
            lines.append(
                f"| {key} | {arm} | {dropped:.3f} | {acc:.4f} | {g} | {lost} | {100 * d:+.2f} | "
                f"[{100 * (d - 1.96 * se):+.2f}, {100 * (d + 1.96 * se):+.2f}] | {p_exact(g, lost):.3f} |"
            )
    lines += ["", "Oracle gains by GQA structural type (largest drop fraction):", ""]
    last = list(rows[0]["drops"])[-1]
    by_type = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        stats = by_type[r["structural"]]
        stats[0] += 1
        stats[1] += r["drops"][last]["oracle"]["correct"] and not r["full"]["correct"]
        stats[2] += r["full"]["correct"] and not r["drops"][last]["oracle"]["correct"]
    lines += ["| type | n | gained | lost |", "| --- | --- | --- | --- |"]
    lines += [f"| {t} | {s[0]} | {s[1]} | {s[2]} |" for t, s in sorted(by_type.items(), key=lambda kv: -kv[1][0])]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="gqa_val_balanced")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "configs" / "splits" / "gqa_val_tune_2k.txt")
    parser.add_argument("--limit", type=int, default=1000, help="First ids of the split (default 1000).")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.25])
    parser.add_argument("--run-id", default="oracle_denoise_tune1k")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "outputs" / "pp3_scorer" / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{args.source}.jsonl"
    previous = finished_rows(path)
    if not args.summary_only:
        baseline = load_yaml(BASELINE_CONFIG)
        model_config = baseline["model"]
        runner = PrunedQwen3VL(
            model_config["id"], dtype=model_config.get("dtype"), device_map=model_config.get("device_map", "auto"),
            attn_implementation=model_config.get("attn_implementation"),
            max_new_tokens=model_config.get("max_new_tokens", 32),
        )
        path.write_text("".join(line + "\n" for line in previous), encoding="utf-8")
        done = {json.loads(line)["sample_id"] for line in previous}
        ids = [i for i in read_split(args.split)[: args.limit] if i not in done]
        source = baseline["data"]["sources"][args.source]
        samples = iter_split_samples(PROJECT_ROOT / source["path"], set(ids), project_root=PROJECT_ROOT)
        generator = torch.Generator().manual_seed(0)
        started_at = utc_now()
        with path.open("a", encoding="utf-8") as handle:
            for sample in tqdm(samples, total=len(ids), desc="oracle"):
                handle.write(json.dumps(process(runner, sample, args.fractions, generator), ensure_ascii=False) + "\n")
                handle.flush()
        write_run_meta(
            run_dir / "run_meta.json", project_root=PROJECT_ROOT, run_id=args.run_id, method="pp3_scorer",
            variant="oracle_denoise", level="oracle", source=args.source, split=str(args.split), limit=args.limit,
            config=f"scripts/pp3/oracle_denoise.py fractions={args.fractions} no fold", dsfold_params=None,
            checkpoint=None, model_id=model_config["id"], started_at=started_at, finished_at=utc_now(),
            rows=len(finished_rows(path)), errors=0,
        )
    rows = [json.loads(line) for line in finished_rows(path)]
    summary = summarize(rows)
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
