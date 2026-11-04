"""Route-B training of the PP3 score head from stored saliency shards (``method.md`` §4.2)."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from token_reduction_vl.pruning.base import BUDGET_LEVELS, budget_for_ratio

from .data import SaliencyRecord, split_by_image
from .losses import distill_loss, listnet_loss, recall_at_k
from .models import build_score_head

TRAIN_LEVELS = ("L1", "L2", "L3", "L4", "L5")


@dataclass(frozen=True)
class RouteBConfig:
    arch: str = "3a"
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 16
    temperature: float = 0.5
    holdout_fraction: float = 0.1
    patience: int = 3
    seed: int = 0
    name: str | None = None
    """Checkpoint name prefix; defaults to ``arch`` (e.g. ``3a_signed`` → ``score_head_3a_signed_seed0.pt``)."""
    loss: str = "listnet"
    """``listnet`` (saliency ranking) or ``distill`` (cross-entropy to a teacher distribution, ADR 0003)."""
    label: str = "primary"
    """Shard target: ``primary``, ``question`` or ``mix`` (see ``SaliencyRecord.target_tensor``);
    ``contrast`` is resolved by scripts/pp3/train_score_head.py via ``contrastive_records``."""
    mix: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _loss(scores: torch.Tensor, target: torch.Tensor, config: RouteBConfig) -> torch.Tensor:
    if config.loss == "listnet":
        return listnet_loss(scores, target, config.temperature)
    if config.loss == "distill":
        return distill_loss(scores, target, config.temperature)
    raise ValueError(f"Unknown loss {config.loss!r}; choose listnet or distill.")


def _evaluate(head: torch.nn.Module, records: list[SaliencyRecord], indices: list[int], config: RouteBConfig) -> dict[str, float]:
    head.eval()
    losses, recalls, baseline_recalls = [], [], []
    ratio = BUDGET_LEVELS["L3"]
    with torch.no_grad():
        for index in indices:
            record = records[index]
            features = record.features(math.log(ratio), config.device)
            saliency = record.target_tensor(config.label, config.mix, config.device)
            scores = head(features)
            k = budget_for_ratio(ratio, features.num_tokens)
            losses.append(float(_loss(scores, saliency, config)))
            recalls.append(recall_at_k(scores, saliency, k))
            # Hand-crafted PP1 quality as the reference ranking (α = 1, β = 0.5 of set D).
            baseline_recalls.append(recall_at_k(features.z_rel + 0.5 * features.z_ms, saliency, k))
    count = max(1, len(indices))
    return {
        # Held-out loss of the configured objective; the key keeps its route-B name for old histories.
        "listnet": sum(losses) / count,
        "recall_at_k_L3": sum(recalls) / count,
        "recall_at_k_L3_handcrafted": sum(baseline_recalls) / count,
    }


def train_route_b(
    records: list[SaliencyRecord],
    dims: tuple[int, int, int],
    config: RouteBConfig,
    checkpoint_dir: Path,
    log_dir: Path,
    *,
    resume: bool = False,
) -> Path:
    """Train with ListNet, early-stop on held-out images, save the best checkpoint.

    ``score_head_<arch>_seed<s>_last.pt`` is written after every epoch; with
    ``resume=True`` training continues from it. Shuffling is seeded per epoch,
    so a resumed run sees the same order as an uninterrupted one.
    """

    torch.manual_seed(config.seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    train_indices, holdout_indices = split_by_image(records, config.holdout_fraction, config.seed)
    head = build_score_head(config.arch, dims).to(config.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.epochs * math.ceil(len(train_indices) / config.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, total_steps))

    best_loss, bad_epochs, history, first_epoch = float("inf"), 0, [], 0
    tag = f"{config.name or config.arch}_seed{config.seed}"
    checkpoint = checkpoint_dir / f"score_head_{tag}.pt"
    last = checkpoint_dir / f"score_head_{tag}_last.pt"
    history_path = log_dir / f"history_{tag}.json"
    if resume and last.exists():
        state = torch.load(last, map_location=config.device, weights_only=False)
        head.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_loss, bad_epochs, history = state["best_loss"], state["bad_epochs"], state["history"]
        first_epoch = state["epoch"] + 1
        print(f"Resuming {tag} from epoch {first_epoch}")
        if bad_epochs >= config.patience:
            return checkpoint

    for epoch in range(first_epoch, config.epochs):
        head.train()
        rng = random.Random(config.seed * 1000 + epoch)
        order = list(train_indices)
        rng.shuffle(order)
        epoch_loss = 0.0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=config.device)
            for index in batch:
                record = records[index]
                # Route B is a ranking target; a random level keeps the log-r input meaningful.
                ratio = BUDGET_LEVELS[rng.choice(TRAIN_LEVELS)]
                features = record.features(math.log(ratio), config.device)
                loss = loss + _loss(head(features), record.target_tensor(config.label, config.mix, config.device), config)
            loss = loss / len(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.detach()) * len(batch)
        metrics = _evaluate(head, records, holdout_indices, config)
        metrics.update({
            "epoch": epoch,
            "train_listnet": epoch_loss / max(1, len(order)),
            "lr": scheduler.get_last_lr()[0],
        })
        history.append(metrics)
        if metrics["listnet"] < best_loss:
            best_loss, bad_epochs = metrics["listnet"], 0
            torch.save({"arch": config.arch, "dims": dims, "state_dict": head.state_dict(), "config": asdict(config)}, checkpoint)
        else:
            bad_epochs += 1
        torch.save(
            {
                "state_dict": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "bad_epochs": bad_epochs,
                "history": history,
            },
            last,
        )
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"[{tag}] epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)))
        if bad_epochs >= config.patience:
            break
    return checkpoint


def load_score_head(checkpoint: Path, device: str = "cpu") -> torch.nn.Module:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    head = build_score_head(state["arch"], tuple(state["dims"]))
    head.load_state_dict(state["state_dict"])
    return head.to(device).eval()
