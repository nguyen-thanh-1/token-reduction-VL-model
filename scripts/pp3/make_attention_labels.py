"""PP3 (ADR 0003): LLM text-to-image attention labels for distilling the score head.

For each sample: the full-token greedy answer (no EOS, <= 8 tokens), one
teacher-forced forward with eager LLM attention, and the attention of the chosen
LLM layers from the answer rows (``saliency``) and the question rows
(``saliency_q``) to the visual tokens. Features are stored as in
make_saliency_labels.py, so train_score_head.py reads the shards unchanged.
Resumable; use --start/--end to split the work.

Examples:
    uv run python scripts/pp3/make_attention_labels.py --end 20
    uv run python scripts/pp3/make_attention_labels.py --end 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from token_reduction_vl.config import load_yaml  # noqa: E402
from token_reduction_vl.data.splits import iter_split_samples, read_split  # noqa: E402
from token_reduction_vl.models.qwen3_vl_pruned import PrunedQwen3VL, question_text_for_scoring  # noqa: E402
from token_reduction_vl.pruning.base import BUDGET_LEVELS, budget_for_ratio  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.attention_teacher import DEFAULT_LAYERS, attention_teacher  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.data import ShardWriter, stored_sample_ids  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.dsfold_base import DSFoldBase, load_dsfold_params  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.features import FeatureCompressor  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.saliency import full_token_answer  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pp3"))
from make_saliency_labels import record_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "methods" / "pp3_scorer_attn.yaml")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    labels = config["labels"]
    model = config["model"]
    layers = tuple(labels.get("layers", DEFAULT_LAYERS))
    print(f"Attention labels: layers {layers[0]}..{layers[-1]} ({len(layers)}), prefix={labels['prefix']}")

    runner = PrunedQwen3VL(model["id"], dtype=model.get("dtype"), device_map=model.get("device_map", "auto"))
    base = DSFoldBase(load_dsfold_params(PROJECT_ROOT / config["dsfold_params"]))
    compressor = FeatureCompressor.load(PROJECT_ROOT / config["compressor"])
    output_dir = PROJECT_ROOT / labels["output_dir"]
    ids = read_split(PROJECT_ROOT / labels["split"])[args.start : args.end]
    skip_log = output_dir / f"{labels['prefix']}_skipped.txt"
    done = stored_sample_ids(output_dir, labels["prefix"])
    if skip_log.exists():
        done |= {line.split("\t")[0] for line in skip_log.read_text(encoding="utf-8").splitlines() if line}
    todo = [sample_id for sample_id in ids if sample_id not in done]
    print(f"{len(ids) - len(todo):,} of {len(ids):,} samples in this range are already done")
    writer = ShardWriter(output_dir, f"{labels['prefix']}_from{args.start:06d}", shard_size=labels.get("shard_size", 500))

    def skip(sample_id: str, reason: str) -> None:
        with skip_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{sample_id}\t{reason}\n")

    # Sink monitor: share of the teacher mass on its single largest token, question-row coverage,
    # and overlap of the two teachers' top-K at L3.
    stats = {"samples": 0, "top1_answer": 0.0, "top1_question": 0.0, "no_question_rows": 0, "overlap_L3": 0.0}
    samples = iter_split_samples(PROJECT_ROOT / labels["canonical"], todo, project_root=PROJECT_ROOT)
    try:
        for sample in tqdm(samples, total=len(todo), desc="attention"):
            text = question_text_for_scoring(sample.question, sample.metadata.get("choices") or [])
            try:
                encoded = runner.encode(sample.image, sample.question, text)
                answer = full_token_answer(runner, encoded)
                if answer.numel() == 0:
                    skip(sample.sample_id, "empty_answer")
                    continue
                teacher = attention_teacher(runner, encoded, answer, layers)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skip(sample.sample_id, "cuda_oom")
                continue
            record_sample(writer, sample, encoded, teacher.answer, runner, base, compressor, saliency_q=teacher.question)
            k = budget_for_ratio(BUDGET_LEVELS["L3"], encoded.num_visual)
            top_a = set(torch.topk(teacher.answer, k).indices.tolist())
            top_q = set(torch.topk(teacher.question, k).indices.tolist())
            stats["samples"] += 1
            stats["top1_answer"] += float(teacher.answer.max() / teacher.answer.sum())
            stats["top1_question"] += float(teacher.question.max() / teacher.question.sum())
            stats["no_question_rows"] += int(teacher.question_rows == 0)
            stats["overlap_L3"] += len(top_a & top_q) / k
    finally:
        writer.flush()
        count = max(1, stats["samples"])
        summary = {key: (value / count if key in ("top1_answer", "top1_question", "overlap_L3") else value)
                   for key, value in stats.items()}
        stats_path = output_dir / f"{labels['prefix']}_from{args.start:06d}_stats.json"
        stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("This run:", json.dumps(summary))
    print(f"Done. Shards in {output_dir}; skipped ids in {skip_log.name}")


if __name__ == "__main__":
    main()
