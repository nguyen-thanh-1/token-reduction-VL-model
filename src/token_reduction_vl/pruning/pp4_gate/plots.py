"""Training curves of the PP4 gate network."""

from __future__ import annotations

import json
from pathlib import Path


def plot_training_log(log_path: Path, output_path: Path | None = None) -> Path:
    """KL, expected keep ratio E[K]/N, auxiliary losses and the λ/κ schedule per step."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{log_path} has no entries yet")
    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        (axes[0, 0], ["total", "kl"], "loss"),
        (axes[0, 1], ["expected_ratio"], "E[K]/N (expected keep ratio)"),
        (axes[1, 0], ["redundancy", "anchor"], "auxiliary terms"),
        (axes[1, 1], ["lambda", "kappa"], "schedule"),
    ]
    for ax, keys, title in panels:
        for key in keys:
            if key in rows[0]:
                ax.plot(steps, [row[key] for row in rows], label=key)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(log_path.stem)
    fig.tight_layout()
    output_path = output_path or log_path.with_suffix(".png")
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
