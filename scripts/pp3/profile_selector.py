"""Per-stage GPU timing of the PP3 selector on synthetic inputs (no VLM weights).

Uses real tensor shapes (N visual tokens, 2048-d, three DeepStack levels) and the
trained score head, so the stage split matches the real selector:
S1 scores, feature build, head forward, DPP, fold.

Example:
    uv run python scripts/pp3/profile_selector.py --method-config configs/methods/pp3_scorer_attn50k.yaml --n 1790 294
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from token_reduction_vl.config import load_yaml  # noqa: E402
from token_reduction_vl.pruning.base import BUDGET_LEVELS, PrunerInput, budget_for_ratio  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.factory import build_pruner  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.features import build_token_features  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.models import quality_from_scores  # noqa: E402

DIM = 2048


def synthetic_input(n: int, ratio: float, device: str, seed: int = 0) -> PrunerInput:
    g = torch.Generator(device="cpu").manual_seed(seed)
    width = max(1, round((n * 4 / 3) ** 0.5))
    height = -(-n // width)
    n = height * width
    rand = lambda *shape: torch.randn(*shape, generator=g).to(device, torch.bfloat16)  # noqa: E731
    embed = torch.nn.Embedding(64, DIM).to(device, torch.bfloat16)
    ids = torch.arange(12, device=device)
    return PrunerInput(
        v=rand(n, DIM), deepstack=[rand(n, DIM) for _ in range(3)], grid_hw=(height, width),
        question_ids=ids, question_tokens=tuple(f"w{i}" for i in range(12)), embed_tokens=embed, ratio=ratio,
    )


def timed(fn, sync):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, 1000 * (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method-config", type=Path, default=PROJECT_ROOT / "configs/methods/pp3_scorer_attn50k.yaml")
    parser.add_argument("--n", type=int, nargs="+", default=[294, 1790])
    parser.add_argument("--levels", nargs="+", default=["L1", "L3", "L4"])
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    pruner = build_pruner(load_yaml(args.method_config), project_root=PROJECT_ROOT, device=device)
    base = pruner.base
    print(f"device={device}  stages in ms, median of {args.repeats}")
    print(f"{'N':>5} {'lvl':>3} {'K':>4} | {'S1':>6} {'feat':>6} {'head':>6} {'dpp':>7} {'fold':>6} | {'total':>7}")
    for n in args.n:
        for level in args.levels:
            x = synthetic_input(n, BUDGET_LEVELS[level], device)
            k = budget_for_ratio(x.ratio, x.num_tokens)
            stages: dict[str, list[float]] = {s: [] for s in ("S1", "feat", "head", "dpp", "fold", "total")}
            for rep in range(args.repeats + 3):
                with torch.no_grad():
                    scores, t1 = timed(lambda: base.scores(x), sync)
                    feats, t2 = timed(lambda: build_token_features(
                        x, scores, stopwords=base.params.stopwords, ratio=x.ratio, compressor=pruner.compressor).to(device), sync)
                    quality, t3 = timed(lambda: quality_from_scores(pruner.head(feats)), sync)
                    sel, t4 = timed(lambda: base.order(scores, k, quality), sync)
                    _, t5 = timed(lambda: base.fold(x, scores, sel.keep_indices()), sync)
                    _, t6 = timed(lambda: pruner.select(x), sync)
                if rep >= 3:
                    for name, t in zip(stages, (t1, t2, t3, t4, t5, t6)):
                        stages[name].append(t)
            med = {s: statistics.median(v) for s, v in stages.items()}
            print(f"{x.num_tokens:>5} {level:>3} {k:>4} | {med['S1']:6.1f} {med['feat']:6.1f} {med['head']:6.1f} "
                  f"{med['dpp']:7.1f} {med['fold']:6.1f} | {med['total']:7.1f}")


if __name__ == "__main__":
    main()
