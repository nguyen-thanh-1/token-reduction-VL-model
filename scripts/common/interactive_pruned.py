"""Interactive pruned inference: pick an image, type a question, see which tokens are kept.

The model and the pruner load once; then each round asks for an image path and
a question. For the full-token model and every level it prints the answer, the
kept / total visual tokens, and a map of the token grid (``#`` kept, ``.``
dropped; with a focus-context config ``F`` focus, ``c`` context). A panel is
saved per round to --output-dir (default data/demo/interactive/, not tracked by
Git): the original image, the score maps (brighter = higher) and the kept
tokens at each level (focus outlined red, context green).

Examples:
    uv run python scripts/common/interactive_pruned.py
    uv run python scripts/common/interactive_pruned.py --level L2 --level L3 --level L4
    uv run python scripts/common/interactive_pruned.py --method-config configs/methods/pp3_scorer_fc.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from token_reduction_vl.config import load_yaml  # noqa: E402
from token_reduction_vl.data.messages import build_question_prompt  # noqa: E402
from token_reduction_vl.data.schema import CanonicalSample, Choice  # noqa: E402
from token_reduction_vl.models.qwen3_vl_pruned import PrunedQwen3VL, question_text_for_scoring  # noqa: E402
from token_reduction_vl.pruning.base import BUDGET_LEVELS, NoOpPruner  # noqa: E402
from token_reduction_vl.reporting.pruning_report import render_kept_tokens  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pruned import load_pruner  # noqa: E402

BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline_qwen3_vl.yaml"
CELL = 32


def build_prompt(question: str, choices: list[str], yes_no: bool) -> tuple[str, list[dict]]:
    labels = "ABCDEFGH"
    options = tuple(Choice(label=labels[i], text=text) for i, text in enumerate(choices))
    task_type = "multiple_choice" if options else ("binary_vqa" if yes_no else "open_vqa")
    sample = CanonicalSample(
        sample_id="demo", dataset="demo", split="demo", task_type=task_type, image="demo",
        image_id="demo", question=question, choices=options,
    )
    return build_question_prompt(sample), [choice.to_dict() for choice in options]


def grid_map(grid_hw: tuple[int, int], keep: list[int], focus: list[int] | None = None) -> str:
    height, width = grid_hw
    kept, focused = set(keep), set(focus or [])

    def mark(index: int) -> str:
        if index in focused:
            return "F"
        if index in kept:
            return "c" if focus is not None else "#"
        return "."

    return "\n".join("    " + "".join(mark(row * width + col) for col in range(width)) for row in range(height))


def render_roles(image: Image.Image, grid_hw: tuple[int, int], keep: list[int], focus: list[int]) -> Image.Image:
    """Kept tokens with focus outlined red and context green; dropped tokens darkened."""

    canvas = render_kept_tokens(image, grid_hw, keep)
    draw = ImageDraw.Draw(canvas)
    width = grid_hw[1]
    for index in focus:
        row, col = divmod(index, width)
        draw.rectangle((col * CELL, row * CELL, (col + 1) * CELL - 1, (row + 1) * CELL - 1), outline=(255, 40, 40), width=3)
    return canvas


def render_quality(image: Image.Image, grid_hw: tuple[int, int], quality: torch.Tensor) -> Image.Image:
    """Image on the merged grid, each token darkened by how low its quality is (rank-normalised)."""

    height, width = grid_hw
    canvas = image.convert("RGB").resize((width * CELL, height * CELL), Image.BICUBIC)
    ranks = torch.argsort(torch.argsort(quality.float())).float() / max(1, quality.numel() - 1)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for index, rank in enumerate(ranks.tolist()):
        row, col = divmod(index, width)
        box = (col * CELL, row * CELL, (col + 1) * CELL - 1, (row + 1) * CELL - 1)
        draw.rectangle(box, fill=(0, 0, 0, int(220 * (1 - rank))))
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")


def panel(tiles: list[tuple[str, Image.Image]], columns: int = 2) -> Image.Image:
    width, height = tiles[0][1].size
    title = 22
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + title)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, tile) in enumerate(tiles):
        x, y = (index % columns) * width, (index // columns) * (height + title)
        draw.text((x + 6, y + 4), label, fill="black")
        sheet.paste(tile.resize((width, height)), (x, y + title))
    return sheet


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip().strip('"')
    except EOFError:
        return "q"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method-config", default="configs/methods/pp3_scorer_attn50k.yaml")
    parser.add_argument("--level", action="append", choices=list(BUDGET_LEVELS), help="Repeatable; default L3 and L4.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "demo" / "interactive")
    parser.add_argument("--no-open", action="store_true", help="Do not open the saved panel.")
    args = parser.parse_args()
    levels = args.level or ["L3", "L4"]

    model = load_yaml(BASELINE_CONFIG)["model"]
    runner = PrunedQwen3VL(
        model["id"], dtype=model.get("dtype"), device_map=model.get("device_map", "auto"),
        attn_implementation=model.get("attn_implementation"), max_new_tokens=model.get("max_new_tokens", 32),
    )
    _, pruner, method_config = load_pruner(args.method_config, str(runner.device))
    head_name = Path(method_config.get("checkpoint") or "dsfold_D").stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nScore head: {head_name}; levels {', '.join(levels)}.")
    print("Each round: image path (Enter = previous image, q = quit), question,")
    print("then options separated by ';' (Enter = none; 'yn' = yes/no question).")

    image_path: Path | None = None
    round_index = 0
    while True:
        typed = ask("\nImage path: ")
        if typed.lower() == "q":
            break
        if typed:
            image_path = Path(typed)
        if image_path is None or not image_path.is_file():
            print(f"  Not a file: {image_path}")
            image_path = None
            continue
        question = ask("Question: ")
        if question.lower() == "q":
            break
        if not question:
            continue
        options = ask("Options: ")
        yes_no = options.lower() == "yn"
        choices = [] if yes_no or not options else [part.strip() for part in options.split(";") if part.strip()]

        image = Image.open(image_path).convert("RGB")
        prompt, choice_dicts = build_prompt(question, choices, yes_no)
        scoring_text = question_text_for_scoring(prompt, choice_dicts)
        encoded = runner.encode(image, prompt, scoring_text)
        n = encoded.num_visual
        height, width = encoded.grid_hw
        print(f"\n  Image {image.width}x{image.height} -> token grid {height}x{width} = {n} visual tokens"
              f" (ViT {encoded.timing['vit']:.3f}s)")

        tiles = [("original", image.resize((width * CELL, height * CELL), Image.BICUBIC))]
        if hasattr(pruner, "token_quality"):
            _, quality = pruner.token_quality(runner.pruner_input(encoded, BUDGET_LEVELS[levels[0]]))
            label = "context quality" if hasattr(pruner, "focus_scores") else "score head quality"
            tiles.append((f"{label} ({head_name})", render_quality(image, encoded.grid_hw, quality.cpu())))
        if hasattr(pruner, "focus_scores"):
            focus_map = pruner.focus_scores(runner.pruner_input(encoded, BUDGET_LEVELS[levels[0]]))
            tiles.append(("focus score (question-specific)", render_quality(image, encoded.grid_hw, focus_map)))

        record = {"image": str(image_path), "question": question, "prompt": prompt, "grid_hw": [height, width],
                  "num_visual": n, "runs": {}}
        for name, current, level in [("full", NoOpPruner(), None), *[(lv, pruner, lv) for lv in levels]]:
            ratio = None if level is None else BUDGET_LEVELS[level]
            result = runner.predict(image, prompt, scoring_text, current, ratio, encoded=encoded)
            input_full = int(encoded.input_ids.shape[1])
            print(f"\n  [{name}] answer: {result.text}")
            print(f"    visual tokens {result.num_kept}/{n} ({100 * result.num_kept / n:.1f}%), "
                  f"LLM input {result.input_tokens}/{input_full} tokens, "
                  f"selector {result.timing['selector']:.3f}s + prefill {result.timing['prefill']:.3f}s "
                  f"+ decode {result.timing['decode']:.3f}s")
            record["runs"][name] = {"answer": result.text, "K": result.num_kept, "N": n,
                                    "input_tokens": result.input_tokens, "keep_indices": result.keep_indices}
            focus = result.pruner_metadata.get("focus_indices")
            if focus is not None:
                record["runs"][name]["focus_indices"] = focus
                print(f"    focus {len(focus)} + context {result.num_kept - len(focus)} tokens "
                      f"(focus share {result.pruner_metadata['focus_share']:.2f})")
            if level is not None:
                legend = "F = focus, c = context" if focus is not None else "# = kept"
                print(f"    kept-token map ({height} rows x {width} cols, {legend}):")
                print(grid_map(encoded.grid_hw, result.keep_indices, focus))
                title = f"{level}: {result.num_kept}/{n} tokens - answer: {result.text}"
                canvas = (render_roles(image, encoded.grid_hw, result.keep_indices, focus) if focus is not None
                          else render_kept_tokens(image, encoded.grid_hw, result.keep_indices))
                tiles.append((title, canvas))
        del encoded

        round_index += 1
        stem = f"{round_index:03d}_{re.sub(r'[^A-Za-z0-9._-]', '_', image_path.stem)}"
        panel_path = args.output_dir / f"{stem}.png"
        panel(tiles).save(panel_path)
        (args.output_dir / f"{stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  Saved {panel_path} (+ .json with kept indices; index = row * {width} + col)")
        if not args.no_open and hasattr(os, "startfile"):
            os.startfile(panel_path)


if __name__ == "__main__":
    main()
