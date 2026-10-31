"""Try a pruning method on one image: print the answers and save kept-token overlays.

The full-token model (NoOp) always runs first for comparison. Give either an
image path plus a question, or a canonical --sample-id (e.g. gqa:testdev:201307251).
Overlays go to --output-dir (default data/demo/, not tracked by Git).

Examples:
    uv run python scripts/common/demo_pruned.py --image path/to/photo.jpg --question "What color is the car?"
    uv run python scripts/common/demo_pruned.py --sample-id gqa:testdev:201307251 --level L1 --level L3 --level L5
    uv run python scripts/common/demo_pruned.py --image x.jpg --question "Which animal is shown?" --choice cat --choice dog
    uv run python scripts/common/demo_pruned.py --sample-id gqa:testdev:201307251 \
        --method-config configs/methods/pp3_scorer.yaml --method-config configs/methods/pp3_scorer_dsfold_D.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from PIL import Image  # noqa: E402

from token_reduction_vl.config import load_yaml  # noqa: E402
from token_reduction_vl.data.messages import build_question_prompt  # noqa: E402
from token_reduction_vl.data.schema import CanonicalSample, Choice  # noqa: E402
from token_reduction_vl.data.splits import iter_split_samples  # noqa: E402
from token_reduction_vl.models.qwen3_vl_pruned import PrunedQwen3VL, question_text_for_scoring  # noqa: E402
from token_reduction_vl.pruning.base import BUDGET_LEVELS, NoOpPruner  # noqa: E402
from token_reduction_vl.reporting.pruning_report import render_kept_tokens  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pruned import load_pruner  # noqa: E402

BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline_qwen3_vl.yaml"


def from_arguments(args: argparse.Namespace):
    """Image, prompt, choices and references for the requested input."""

    if args.sample_id:
        dataset, split = args.sample_id.split(":")[:2]
        canonical = PROJECT_ROOT / "data" / "processed" / "v1" / dataset / f"{split}.jsonl"
        samples = list(iter_split_samples(canonical, {args.sample_id}, project_root=PROJECT_ROOT))
        if not samples:
            raise SystemExit(f"{args.sample_id} not found in {canonical}")
        sample = samples[0]
        return sample.image, sample.question, sample.metadata.get("choices") or [], list(sample.answers)
    if not (args.image and args.question):
        raise SystemExit("Give --image and --question, or --sample-id.")
    labels = "ABCDEFGH"
    choices = tuple(Choice(label=labels[i], text=text) for i, text in enumerate(args.choice or []))
    task_type = "multiple_choice" if choices else ("binary_vqa" if args.yes_no else "open_vqa")
    sample = CanonicalSample(
        sample_id="demo", dataset="demo", split="demo", task_type=task_type, image=str(args.image),
        image_id="demo", question=args.question, choices=choices,
    )
    return Image.open(args.image).convert("RGB"), build_question_prompt(sample), [c.to_dict() for c in choices], []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--choice", action="append", help="Repeatable; makes it a multiple-choice question.")
    parser.add_argument("--yes-no", action="store_true", help="Ask for a yes/no answer.")
    parser.add_argument("--sample-id", default=None, help="Canonical id, e.g. gqa:testdev:201307251.")
    parser.add_argument("--method-config", action="append", help="Repeatable; default configs/methods/pp3_scorer.yaml.")
    parser.add_argument("--level", action="append", choices=list(BUDGET_LEVELS), help="Repeatable; default L3.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "demo")
    args = parser.parse_args()
    method_configs = args.method_config or ["configs/methods/pp3_scorer.yaml"]
    levels = args.level or ["L3"]

    image, prompt, choices, references = from_arguments(args)
    model = load_yaml(BASELINE_CONFIG)["model"]
    runner = PrunedQwen3VL(
        model["id"], dtype=model.get("dtype"), device_map=model.get("device_map", "auto"),
        attn_implementation=model.get("attn_implementation"), max_new_tokens=model.get("max_new_tokens", 32),
    )
    scoring_text = question_text_for_scoring(prompt, choices)
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", args.sample_id or args.image.stem)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPrompt:\n{prompt}\n" + (f"Reference: {', '.join(references)}\n" if references else ""))
    runs = [("full_tokens", NoOpPruner(), None)]
    for config in method_configs:
        method, pruner, method_config = load_pruner(config, str(runner.device))
        variant = Path(method_config.get("checkpoint") or "dsfold_D").stem
        runs += [(f"{method}_{variant}_{level}", pruner, level) for level in levels]

    print(f"{'run':<45} {'K/N':>11} {'input':>6} {'latency':>8}  answer")
    for name, pruner, level in runs:
        result = runner.predict(image, prompt, scoring_text, pruner, None if level is None else BUDGET_LEVELS[level])
        print(f"{name:<45} {result.num_kept:>5}/{result.num_visual:<5} {result.input_tokens:>6} {result.latency_seconds:>7.3f}s  {result.text}")
        if level is not None:
            path = args.output_dir / f"{stem}__{name}.png"
            render_kept_tokens(image, result.grid_hw, result.keep_indices).save(path)
    print(f"\nKept-token overlays (green = kept, dark = dropped) in {args.output_dir}")


if __name__ == "__main__":
    main()
