"""Training curves of the PP3 score head (route B)."""

from __future__ import annotations

import json
from pathlib import Path


def plot_history(history_path: Path, output_path: Path | None = None) -> Path:
    """ListNet loss (train vs. held-out) and recall@K at L3 (head vs. hand-crafted) per epoch."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [entry["epoch"] for entry in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].plot(epochs, [e["train_listnet"] for e in history], marker="o", label="train")
    axes[0].plot(epochs, [e["listnet"] for e in history], marker="o", label="held-out images")
    axes[0].set_title("ListNet loss")
    axes[1].plot(epochs, [e["recall_at_k_L3"] for e in history], marker="o", label="score head")
    axes[1].plot(epochs, [e["recall_at_k_L3_handcrafted"] for e in history], linestyle="--", label="PP1 quality (set D)")
    axes[1].set_title("held-out recall@K of top-saliency tokens (L3)")
    axes[2].plot(epochs, [e.get("lr", float("nan")) for e in history], marker="o")
    axes[2].set_title("learning rate")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(history_path.stem)
    fig.tight_layout()
    output_path = output_path or history_path.with_suffix(".png")
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
