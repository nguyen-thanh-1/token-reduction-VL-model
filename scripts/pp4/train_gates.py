"""PP4 steps 5-6: train the gate network for one lambda through the frozen Qwen3-VL.

Use --max-samples for the pilot run (README step 5) before the full run.

Example:
    uv run python scripts/pp4/train_gates.py --lambda-target 1.0 --max-samples 5000
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
from token_reduction_vl.pruning.pp4_gate.dsfold_base import DSFoldBase, load_dsfold_params  # noqa: E402
from token_reduction_vl.pruning.pp4_gate.losses import GateSchedule  # noqa: E402
from token_reduction_vl.pruning.pp4_gate.model import GateNet  # noqa: E402
from token_reduction_vl.pruning.pp4_gate.train import (  # noqa: E402
    GateTrainConfig,
    TeacherTarget,
    save_gate_net,
    train_step,
)


def load_teacher(path: Path) -> dict[str, TeacherTarget]:
    data = np.load(path)
    targets = {}
    for index, sample_id in enumerate(json.loads(str(data["sample_ids"]))):
        answer = data["answers"][index]
        length = int((answer >= 0).sum())
        targets[sample_id] = TeacherTarget(
            answer_ids=torch.from_numpy(answer[:length].astype(np.int64)),
            topk_ids=torch.from_numpy(data["topk_ids"][index][:length].astype(np.int64)),
            topk_logprobs=torch.from_numpy(data["topk_logprobs"][index][:length].astype(np.float32)),
        )
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "methods" / "pp4_gate.yaml")
    parser.add_argument("--lambda-target", type=float, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    training = dict(config["training"])
    output_dir = PROJECT_ROOT / training.pop("output_dir")
    training.pop("lambdas", None)
    training.pop("pilot_samples", None)
    train_config = GateTrainConfig(lambda_target=args.lambda_target, **training)
    model = config["model"]

    torch.manual_seed(train_config.seed)
    runner = PrunedQwen3VL(
        model["id"], dtype=model.get("dtype"), device_map=model.get("device_map", "auto"),
        attn_implementation=model.get("attn_implementation"),
    )
    if not args.no_grad_checkpointing:
        lm = runner.language_model
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        lm.train()  # checkpointing only runs in training mode; Qwen3-VL has no dropout
    base = DSFoldBase(load_dsfold_params(PROJECT_ROOT / config["dsfold_params"]))
    net = GateNet().to(runner.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=train_config.lr)

    teacher = load_teacher(PROJECT_ROOT / config["teacher"]["output"])
    ids = [sample_id for sample_id in read_split(PROJECT_ROOT / config["teacher"]["split"]) if sample_id in teacher]
    ids = ids[: args.max_samples]
    total_steps = train_config.epochs * len(ids)
    schedule = GateSchedule(total_steps=total_steps, lambda_target=train_config.lambda_target)
    tag = f"lambda{train_config.lambda_target:g}" + ("_pilot" if args.max_samples else "")
    log_path = output_dir / "training" / f"{tag}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(train_config.epochs):
            samples = iter_split_samples(PROJECT_ROOT / config["teacher"]["canonical"], ids, project_root=PROJECT_ROOT)
            optimizer.zero_grad(set_to_none=True)
            for sample in tqdm(samples, total=len(ids), desc=f"{tag} epoch {epoch}"):
                text = question_text_for_scoring(sample.question, sample.metadata.get("choices") or [])
                encoded = runner.encode(sample.image, sample.question, text)
                lam, kappa = schedule.weights(step)
                terms = train_step(runner, net, base, encoded, teacher[sample.sample_id], lam=lam, kappa=kappa, config=train_config)
                (terms.total / train_config.grad_accumulation).backward()
                step += 1
                if step % train_config.grad_accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step % 50 == 0:
                    log.write(json.dumps({
                        "step": step, "epoch": epoch, "lambda": lam, "kappa": kappa, "kl": terms.kl,
                        "expected_ratio": terms.expected_ratio, "redundancy": terms.redundancy, "anchor": terms.anchor,
                    }) + "\n")
                    log.flush()
            save_gate_net(net, output_dir / "checkpoints" / f"gate_{tag}_epoch{epoch}.pt", config=train_config, step=step)
    save_gate_net(net, output_dir / "checkpoints" / f"gate_{tag}.pt", config=train_config, step=step)
    print(f"Finished {step} steps; logs in {log_path}")


if __name__ == "__main__":
    main()
