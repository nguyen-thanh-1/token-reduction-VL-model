"""Run one pruning method on one benchmark source at one budget level.

Writes outputs/<method>/runs/<run_id>/<source>.jsonl and run_meta.json
(METHODS.md section 9). Use --method-config none to run the NoOp pruner.

Example:
    uv run python scripts/common/run_pruned.py --method-config configs/methods/pp3_scorer.yaml \
        --source gqa_testdev_balanced --level L3 --split configs/splits/dev_1k_gqa.txt --limit 10
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tqdm import tqdm  # noqa: E402

from token_reduction_vl.config import load_yaml  # noqa: E402
from token_reduction_vl.data.sources import iter_source_samples  # noqa: E402
from token_reduction_vl.data.splits import read_split  # noqa: E402
from token_reduction_vl.evaluation.run_meta import utc_now, write_run_meta  # noqa: E402
from token_reduction_vl.evaluation.runner import normalize_answer  # noqa: E402
from token_reduction_vl.models.qwen3_vl_pruned import PrunedQwen3VL, question_text_for_scoring  # noqa: E402
from token_reduction_vl.pruning.base import BUDGET_LEVELS, NoOpPruner  # noqa: E402

BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline_qwen3_vl.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method-config", required=True, help="configs/methods/<pp>.yaml or 'none'.")
    parser.add_argument("--source", required=True, help="Source name from configs/baseline_qwen3_vl.yaml.")
    parser.add_argument("--level", required=True, choices=[*BUDGET_LEVELS, "adaptive"])
    parser.add_argument("--split", type=Path, default=None, help="Optional configs/splits/*.txt id list.")
    parser.add_argument("--variant", default="default")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def load_pruner(method_config: str, device: str):
    if method_config == "none":
        return "none", NoOpPruner(), {}
    config = load_yaml(PROJECT_ROOT / method_config)
    module_name, function_name = config["factory"].split(":")
    factory = getattr(importlib.import_module(module_name), function_name)
    return config["method"], factory(config, project_root=PROJECT_ROOT, device=device), config


def main() -> None:
    args = parse_args()
    baseline = load_yaml(BASELINE_CONFIG)
    model_config = baseline["model"]
    runner = PrunedQwen3VL(
        model_config["id"],
        dtype=model_config.get("dtype"),
        device_map=model_config.get("device_map", "auto"),
        attn_implementation=model_config.get("attn_implementation"),
        max_new_tokens=model_config.get("max_new_tokens", 32),
    )
    method, pruner, method_config = load_pruner(args.method_config, str(runner.device))
    ratio = None if args.level == "adaptive" else BUDGET_LEVELS[args.level]
    split_name = args.split.stem if args.split else "full"
    run_id = args.run_id or f"{date.today():%Y%m%d}_{method}_{args.variant}_{args.level}_{split_name}"
    run_dir = PROJECT_ROOT / "outputs" / method / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = iter_source_samples(baseline["data"]["sources"][args.source], project_root=PROJECT_ROOT)
    wanted = set(read_split(args.split)) if args.split else None
    started_at = utc_now()
    count = 0
    output_path = run_dir / f"{args.source}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in tqdm(samples, desc=f"{method}:{args.level}"):
            if wanted is not None and sample.sample_id not in wanted:
                continue
            if args.limit is not None and count >= args.limit:
                break
            scoring_text = question_text_for_scoring(sample.question, sample.metadata.get("choices") or [])
            result = runner.predict(sample.image, sample.question, scoring_text, pruner, ratio)
            task_type = sample.metadata.get("task_type")
            references = list(sample.answers)
            record = {
                "sample_id": sample.sample_id,
                "dataset": sample.metadata.get("dataset"),
                "split": sample.metadata.get("split"),
                "task_type": task_type,
                "image_id": sample.metadata.get("image_id"),
                "question": sample.question,
                "references": references,
                "prediction": result.text,
                "exact_match": (
                    any(normalize_answer(result.text) == normalize_answer(answer) for answer in references)
                    if references and task_type in (None, "open_vqa")
                    else None
                ),
                "method": method,
                "variant": args.variant,
                "level": args.level,
                "ratio_target": ratio,
                "N": result.num_visual,
                "K": result.num_kept,
                "keep_ratio": result.num_kept / result.num_visual,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_seconds": result.latency_seconds,
                "timing": result.timing,
                "peak_vram_mb": result.peak_vram_mb,
                "pruning_method": method,
                "pruner_metadata": result.pruner_metadata,
                "metadata": sample.metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    write_run_meta(
        run_dir / "run_meta.json",
        project_root=PROJECT_ROOT,
        run_id=run_id,
        method=method,
        variant=args.variant,
        level=args.level,
        source=args.source,
        split=str(args.split) if args.split else "full",
        limit=args.limit,
        config=args.method_config,
        dsfold_params=method_config.get("dsfold_params"),
        checkpoint=method_config.get("checkpoint"),
        model_id=model_config["id"],
        started_at=started_at,
        finished_at=utc_now(),
        rows=count,
    )
    print(f"Wrote {count:,} predictions to {output_path}")


if __name__ == "__main__":
    main()
