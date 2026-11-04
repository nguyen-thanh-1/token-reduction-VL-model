"""PP4 step 2: check that a custom 4D float attention bias works (method.md section 4.1).

Checks on one sample:
  1. bias == 0 gives the same logits as no mask;
  2. a large negative bias on half of the image columns is close to deleting those tokens;
  3. the gradient with respect to the gates is non-zero;
  4. the eps = 1e-4 bias used in training, for reference.
Writes outputs/pp4_gate/spike/result.json.

Example:
    uv run python scripts/pp4/spike_attention_bias.py --sample-index 0
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

from token_reduction_vl.config import load_yaml  # noqa: E402
from token_reduction_vl.data.canonical import iter_canonical_samples  # noqa: E402
from token_reduction_vl.models.qwen3_vl_pruned import PrunedQwen3VL, question_text_for_scoring  # noqa: E402
from token_reduction_vl.pruning.base import PrunerOutput  # noqa: E402
from token_reduction_vl.pruning.pp4_gate.attention_bias import gated_attention_mask  # noqa: E402


def full_selection(encoded, deepstack=None) -> PrunerOutput:
    n = encoded.num_visual
    return PrunerOutput(
        keep_indices=torch.arange(n, device=encoded.v.device),
        v_kept=encoded.v,
        deepstack_kept=list(deepstack if deepstack is not None else encoded.deepstack),
        metadata={"K": n, "N": n, "method": "spike", "ratio_target": 1.0},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "methods" / "pp4_gate.yaml")
    parser.add_argument("--canonical", type=Path, default=PROJECT_ROOT / "data" / "processed" / "v1" / "gqa" / "testdev.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    model = load_yaml(args.config)["model"]
    runner = PrunedQwen3VL(
        model["id"], dtype=model.get("dtype"), device_map=model.get("device_map", "auto"),
        attn_implementation=model.get("attn_implementation"),
    )
    for index, sample in enumerate(iter_canonical_samples(args.canonical, project_root=PROJECT_ROOT)):
        if index == args.sample_index:
            break
    text = question_text_for_scoring(sample.question, sample.metadata.get("choices") or [])
    encoded = runner.encode(sample.image, sample.question, text)
    prepared = runner.prepare(encoded, full_selection(encoded))
    answer, _, _ = runner.greedy_decode(prepared)
    answer_ids = torch.tensor([t for t in answer if t not in runner.eos_token_ids][:8] or answer[:1], device=runner.device)
    total = prepared.seq_len + int(answer_ids.numel())
    dtype = prepared.inputs_embeds.dtype
    n = encoded.num_visual
    result: dict[str, object] = {"sample_id": sample.sample_id, "N": n, "attn_implementation": model.get("attn_implementation")}

    with torch.no_grad():
        reference = runner.teacher_forced_logits(prepared, answer_ids)
        ones = torch.ones(n, device=runner.device)
        zero_bias = gated_attention_mask(total, encoded.image_positions, ones, eps=0.0, dtype=dtype)
        result["check1_max_abs_diff_zero_bias"] = float((runner.teacher_forced_logits(prepared, answer_ids, attention_mask=zero_bias) - reference).abs().max())

        gates = ones.clone()
        dropped = torch.arange(0, n, 2, device=runner.device)
        gates[dropped] = 0.0
        hard_bias = gated_attention_mask(total, encoded.image_positions, gates, eps=1e-30, dtype=dtype)
        biased = runner.teacher_forced_logits(prepared, answer_ids, attention_mask=hard_bias)
        keep = torch.arange(1, n, 2, device=runner.device)
        deleted = runner.prepare(
            encoded,
            PrunerOutput(keep, encoded.v[keep], [level[keep] for level in encoded.deepstack],
                         {"K": int(keep.numel()), "N": n, "method": "spike", "ratio_target": 0.5}),
        )
        removed = runner.teacher_forced_logits(deleted, answer_ids)
        kl = torch.nn.functional.kl_div(torch.log_softmax(biased, -1), torch.log_softmax(removed, -1), log_target=True, reduction="batchmean")
        result["check2_kl_bias_vs_deletion"] = float(kl)
        result["check2_argmax_agree"] = bool((biased.argmax(-1) == removed.argmax(-1)).all())

        soft_bias = gated_attention_mask(total, encoded.image_positions, gates, eps=1e-4, dtype=dtype)
        soft = runner.teacher_forced_logits(prepared, answer_ids, attention_mask=soft_bias)
        result["check4_kl_eps1e-4_vs_deletion"] = float(
            torch.nn.functional.kl_div(torch.log_softmax(soft, -1), torch.log_softmax(removed, -1), log_target=True, reduction="batchmean")
        )

    learnable = torch.full((n,), 0.5, device=runner.device, requires_grad=True)
    with torch.enable_grad():
        mask = gated_attention_mask(total, encoded.image_positions, learnable, eps=1e-4, dtype=dtype)
        logits = runner.teacher_forced_logits(prepared, answer_ids, attention_mask=mask)
        log_prob = torch.log_softmax(logits, -1).gather(1, answer_ids.view(-1, 1)).sum()
        (grad,) = torch.autograd.grad(log_prob, learnable)
    result["check3_grad_norm"] = float(grad.norm())
    result["passed"] = bool(
        result["check1_max_abs_diff_zero_bias"] < 1e-2 and result["check2_argmax_agree"] and result["check3_grad_norm"] > 0
    )

    output = PROJECT_ROOT / "outputs" / "pp4_gate" / "spike" / "result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
