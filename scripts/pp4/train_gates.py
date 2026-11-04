"""PP4 steps 5-6: train the gate network for one lambda through the frozen Qwen3-VL.

Use --max-samples for the pilot run (README step 5) before the full run.
A resumable checkpoint gate_<tag>_last.pt is written every `save_every` steps
and at the end of each epoch; pass --resume to continue after an interruption.

Example:
    uv run python scripts/pp4/train_gates.py --lambda-target 1.0 --max-samples 5000
    uv run python scripts/pp4/train_gates.py --lambda-target 1.0 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
from token_reduction_vl.pruning.pp4_gate.dsfold_base import DSFoldBase, load_dsfold_params  # noqa: E402
from token_reduction_vl.pruning.pp4_gate.losses import GateSchedule  # noqa: E402
from token_reduction_vl.pruning.pp4_gate.model import GateNet  # noqa: E402
from token_reduction_vl.pruning.pp4_gate.teacher import load_teacher_targets  # noqa: E402
from token_reduction_vl.pruning.pp4_gate.train import GateTrainConfig, save_gate_net, train_step  # noqa: E402


def save_last(path: Path, net, optimizer, *, step: int, epoch: int, position: int, config: GateTrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "position": position,
            "config": asdict(config),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "methods" / "pp4_gate.yaml")
    parser.add_argument("--lambda-target", type=float, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    training = dict(config["training"])
    output_dir = PROJECT_ROOT / training.pop("output_dir")
    save_every = int(training.pop("save_every", 500))
    training.pop("lambdas", None)
    training.pop("pilot_samples", None)
    train_config = GateTrainConfig(lambda_target=args.lambda_target, **training)
    model = config["model"]
    teacher_cfg = config["teacher"]

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

    teacher = load_teacher_targets(PROJECT_ROOT / teacher_cfg["output_dir"], teacher_cfg["prefix"])
    if not teacher:
        raise SystemExit("No teacher shards found; run scripts/pp4/make_teacher_logits.py first.")
    ids = [sample_id for sample_id in read_split(PROJECT_ROOT / teacher_cfg["split"]) if sample_id in teacher]
    ids = ids[: args.max_samples]
    schedule = GateSchedule(total_steps=train_config.epochs * len(ids), lambda_target=train_config.lambda_target)
    tag = f"lambda{train_config.lambda_target:g}" + ("_pilot" if args.max_samples else "")
    log_path = output_dir / "training" / f"{tag}.jsonl"
    errors_path = output_dir / "training" / f"{tag}_errors.jsonl"
    last_path = output_dir / "checkpoints" / f"gate_{tag}_last.pt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    step, first_epoch, position = 0, 0, 0
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=runner.device, weights_only=False)
        net.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        step, first_epoch, position = state["step"], state["epoch"], state["position"]
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resuming {tag}: step {step}, epoch {first_epoch}, {position} samples into the epoch")
    elif log_path.exists():
        log_path.unlink()  # a fresh run starts a fresh log

    with log_path.open("a", encoding="utf-8") as log:
        for epoch in range(first_epoch, train_config.epochs):
            samples = iter_split_samples(PROJECT_ROOT / teacher_cfg["canonical"], ids, project_root=PROJECT_ROOT)
            optimizer.zero_grad(set_to_none=True)
            skip = position if epoch == first_epoch else 0
            for index, sample in enumerate(tqdm(samples, total=len(ids), desc=f"{tag} epoch {epoch}")):
                if index < skip:
                    continue
                text = question_text_for_scoring(sample.question, sample.metadata.get("choices") or [])
                lam, kappa = schedule.weights(step)
                try:
                    encoded = runner.encode(sample.image, sample.question, text)
                    terms = train_step(
                        runner, net, base, encoded, teacher[sample.sample_id], lam=lam, kappa=kappa, config=train_config
                    )
                    (terms.total / train_config.grad_accumulation).backward()
                except torch.cuda.OutOfMemoryError as exc:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    with errors_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"step": step, "sample_id": sample.sample_id, "error": str(exc)[:300]}) + "\n")
                    terms = None
                step += 1
                if step % train_config.grad_accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if step % save_every == 0:
                        save_last(last_path, net, optimizer, step=step, epoch=epoch, position=index + 1, config=train_config)
                if terms is not None and step % 50 == 0:
                    log.write(json.dumps({
                        "step": step, "epoch": epoch, "lambda": lam, "kappa": kappa, "total": float(terms.total.detach()),
                        "kl": terms.kl, "expected_ratio": terms.expected_ratio,
                        "redundancy": terms.redundancy, "anchor": terms.anchor,
                    }) + "\n")
                    log.flush()
            save_gate_net(net, output_dir / "checkpoints" / f"gate_{tag}_epoch{epoch}.pt", config=train_config, step=step)
            save_last(last_path, net, optimizer, step=step, epoch=epoch + 1, position=0, config=train_config)
    save_gate_net(net, output_dir / "checkpoints" / f"gate_{tag}.pt", config=train_config, step=step)
    print(f"Finished {step} steps; logs in {log_path}")


if __name__ == "__main__":
    main()
