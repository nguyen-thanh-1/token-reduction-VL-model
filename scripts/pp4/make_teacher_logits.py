"""PP4 step 4: full-token teacher answers and top-k log-probabilities for the training split.

Stores, per sample, the greedy answer (<= 8 tokens, no EOS) and the top-k
log-probabilities at every answer position (teacher-forced).

Example:
    uv run python scripts/pp4/make_teacher_logits.py --limit 100
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from token_reduction_vl.config import load_yaml  # noqa: E402
from token_reduction_vl.data.splits import iter_split_samples, read_split  # noqa: E402
from token_reduction_vl.models.qwen3_vl_pruned import PrunedQwen3VL, question_text_for_scoring  # noqa: E402
from token_reduction_vl.pruning.base import PrunerOutput  # noqa: E402

MAX_ANSWER = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "methods" / "pp4_gate.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    teacher = config["teacher"]
    model = config["model"]
    top_k = int(teacher["top_k"])
    runner = PrunedQwen3VL(model["id"], dtype=model.get("dtype"), device_map=model.get("device_map", "auto"))
    ids = read_split(PROJECT_ROOT / teacher["split"])[: args.limit]

    sample_ids, answers, topk_ids, topk_logprobs = [], [], [], []
    samples = iter_split_samples(PROJECT_ROOT / teacher["canonical"], ids, project_root=PROJECT_ROOT)
    for sample in tqdm(samples, total=len(ids), desc="teacher"):
        text = question_text_for_scoring(sample.question, sample.metadata.get("choices") or [])
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
            continue
        answer_ids = torch.tensor(answer, device=runner.device)
        with torch.no_grad():
            log_probs = torch.log_softmax(runner.teacher_forced_logits(prepared, answer_ids), dim=-1)
        values, indices = torch.topk(log_probs, top_k, dim=-1)
        padded_ids = np.full((MAX_ANSWER, top_k), -1, dtype=np.int32)
        padded_lp = np.full((MAX_ANSWER, top_k), -np.inf, dtype=np.float16)
        padded_ids[: len(answer)] = indices.cpu().numpy()
        padded_lp[: len(answer)] = values.cpu().numpy().astype(np.float16)
        sample_ids.append(sample.sample_id)
        answers.append(answer + [-1] * (MAX_ANSWER - len(answer)))
        topk_ids.append(padded_ids)
        topk_logprobs.append(padded_lp)

    output_path = PROJECT_ROOT / teacher["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        sample_ids=np.array(json.dumps(sample_ids)),
        answers=np.array(answers, dtype=np.int32),
        topk_ids=np.stack(topk_ids),
        topk_logprobs=np.stack(topk_logprobs),
    )
    print(f"Saved {len(sample_ids):,} teacher targets to {output_path}")


if __name__ == "__main__":
    main()
