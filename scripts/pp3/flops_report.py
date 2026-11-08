"""Analytic FLOPs of the LLM prefill and of the selector, per run (CPU only, from run files).

Counts multiply-adds as 2 FLOPs. The LLM is Qwen3-VL-2B's text decoder (read
from the model config); the ViT is identical for every method and is excluded.

    prefill  = Σ_layers [2T·d·(2·H·h + 2·G·h) + 6T·d·m + 2·H·h·T(T+1)]  + 2·d·V
    selector = 2N²·D (cosine matrix) + 2N·K² (greedy DPP)  [+ head]
    head     ≈ 2·P·N + 4N²·256 (one self-attention layer over the N tokens)

T = LLM input tokens of the sample, N/K = visual tokens before/after selection,
P = score-head parameters, D = 2048. Max–min (N > n_cap) costs 2N·K·D instead.

Example:
    uv run python scripts/pp3/flops_report.py --runs random_L3 divprune_L3 cdpruner_L3 3d_attn50k_L1 3d_attn50k_L3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from token_reduction_vl.reporting.pruning_report import read_jsonl  # noqa: E402
from final_report import BENCHMARKS, run_path  # noqa: E402

TEXT = dict(d=2048, m=6144, layers=28, heads=16, kv_heads=8, head_dim=128, vocab=151936)
VISUAL_DIM = 2048
N_CAP = 4096
HEAD_PARAMS = 1_993_153
HEAD_DIM = 256


def text_config(path: Path | None) -> dict[str, int]:
    if path is None or not path.is_file():
        return TEXT
    t = json.loads(path.read_text(encoding="utf-8")).get("text_config", {})
    return dict(d=t["hidden_size"], m=t["intermediate_size"], layers=t["num_hidden_layers"],
                heads=t["num_attention_heads"], kv_heads=t["num_key_value_heads"], head_dim=t["head_dim"],
                vocab=t["vocab_size"])


def prefill_flops(tokens: int, c: dict[str, int]) -> float:
    d, h = c["d"], c["head_dim"]
    projections = 2 * tokens * d * (2 * c["heads"] * h + 2 * c["kv_heads"] * h)
    mlp = 6 * tokens * d * c["m"]
    attention = 2 * c["heads"] * h * tokens * (tokens + 1)  # causal QKᵀ and AV
    return c["layers"] * (projections + mlp + attention) + 2 * d * c["vocab"]


def selector_flops(n: int, k: int, method: str) -> float:
    if method in ("full", "random"):
        return 0.0
    if method == "fastv":
        return float("nan")  # one extra partial prefill; not comparable, see paper
    choose = 2 * n * k * VISUAL_DIM if n > N_CAP else 2 * n * n * VISUAL_DIM + 2 * n * k * k
    head = 2 * HEAD_PARAMS * n + 4 * n * n * HEAD_DIM if method in ("3d_attn50k", "fcd25") else 0.0
    return choose + head


def method_of(name: str) -> str:
    return name if name == "full" else name.rsplit("_", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", default="final")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--model-config", type=Path, default=None, help="Qwen3-VL config.json (default: built-in 2B values)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    c = text_config(args.model_config)

    lines = ["| Run | " + " | ".join(f"{d.upper()} prefill TFLOPs | {d.upper()} selector GFLOPs | {d.upper()} speed-up" for d, _ in BENCHMARKS) + " |",
             "|---|" + "---|---|---|" * len(BENCHMARKS)]
    full_prefill: dict[str, float] = {}
    for name in ["full", *args.runs]:
        cells = []
        for dataset, source in BENCHMARKS:
            path = run_path(args.prefix, name, dataset, source)
            if not path.is_file():
                cells += ["-", "-", "-"]
                continue
            rows = read_jsonl(path)
            pre = statistics.mean(prefill_flops(r["input_tokens"], c) for r in rows)
            sel = statistics.mean(selector_flops(r["N"], r["K"], method_of(name)) for r in rows)
            if name == "full":
                full_prefill[dataset] = pre
            ratio = full_prefill[dataset] / (pre + (0 if sel != sel else sel)) if dataset in full_prefill else float("nan")
            cells += [f"{pre / 1e12:.2f}", "-" if sel != sel else f"{sel / 1e9:.1f}", f"{ratio:.2f}"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
