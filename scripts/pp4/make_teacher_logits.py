"""PP4 step 4: full-token teacher answers and top-k log-probabilities for the training split.

Stores, per sample, the greedy answer (<= 8 tokens, no EOS) and the top-k
log-probabilities at every answer position (teacher-forced), in shards under
data/features/pp4_gate/. Re-running the same command resumes: ids already
stored (or skipped before) are not recomputed. Ctrl+C keeps the buffered rows.

Example:
    uv run python scripts/pp4/make_teacher_logits.py --start 0 --end 10000
"""

from __future__ import annotations

import argparse
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
from token_reduction_vl.pruning.base import PrunerOutput  # noqa: E402
from token_reduction_vl.pruning.pp4_gate.teacher import MAX_ANSWER, TeacherShardWriter, stored_teacher_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "methods" / "pp4_gate.yaml")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    teacher = config["teacher"]
    model = config["model"]
    top_k = int(teacher["top_k"])
    output_dir = PROJECT_ROOT / teacher["output_dir"]
    skip_log = output_dir / f"{teacher['prefix']}_skipped.txt"

    ids = read_split(PROJECT_ROOT / teacher["split"])[args.start : args.end]
    done = stored_teacher_ids(output_dir, teacher["prefix"])
    if skip_log.exists():
        done |= {line.split("\t")[0] for line in skip_log.read_text(encoding="utf-8").splitlines() if line}
    todo = [sample_id for sample_id in ids if sample_id not in done]
    print(f"{len(ids) - len(todo):,} of {len(ids):,} samples in this range are already done")

    runner = PrunedQwen3VL(model["id"], dtype=model.get("dtype"), device_map=model.get("device_map", "auto"))
    writer = TeacherShardWriter(
        output_dir, f"{teacher['prefix']}_from{args.start:06d}", top_k, shard_size=teacher.get("shard_size", 500)
    )

    def skip(sample_id: str, reason: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with skip_log.open("a", encoding="utf-8") as handle:
            handle.write(f"{sample_id}\t{reason}\n")

    samples = iter_split_samples(PROJECT_ROOT / teacher["canonical"], todo, project_root=PROJECT_ROOT)
    try:
        for sample in tqdm(samples, total=len(todo), desc="teacher"):
            text = question_text_for_scoring(sample.question, sample.metadata.get("choices") or [])
            try:
                encoded = runner.encode(sample.image, sample.question, text)
                n = encoded.num_visual
                prepared = runner.prepare(
                    encoded,
                    PrunerOutput(torch.arange(n, device=runner.device), encoded.v, list(encoded.deepstack),
                                 {"K": n, "N": n, "method": "teacher", "ratio_target": 1.0}),
                )
                output, _, _ = runner.greedy_decode(prepared)
                answer = [token for token in output if token not in runner.eos_token_ids][:MAX_ANSWER]
                if not answer:
                    skip(sample.sample_id, "empty_answer")
                    continue
                with torch.no_grad():
                    logits = runner.teacher_forced_logits(prepared, torch.tensor(answer, device=runner.device))
                    values, indices = torch.topk(torch.log_softmax(logits, dim=-1), top_k, dim=-1)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skip(sample.sample_id, "cuda_oom")
                continue
            writer.add(sample.sample_id, answer, indices.cpu().numpy(), values.cpu().numpy())
    finally:
        writer.flush()
    print(f"Done. Shards in {output_dir}")


if __name__ == "__main__":
    main()
