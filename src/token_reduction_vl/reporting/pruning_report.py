"""Shared reporting for pruned runs (``METHODS.md`` §9): summary tables, plots, kept-token maps.

Every run is compared with the full-token baseline on *exactly the same*
sample ids, so dev-subset runs are not compared against full-benchmark numbers.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from token_reduction_vl.evaluation.metrics import score_prediction_record, score_prediction_records

MME_SIZE_BUCKETS = ((0, 512, "N<=512"), (512, 4096, "512<N<=4096"), (4096, math.inf, "N>4096"))
TIMING_KEYS = ("vit", "selector", "prefill", "decode")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quality(dataset: str, rows: list[Mapping[str, Any]]) -> dict[str, float | None]:
    metrics = score_prediction_records(rows)
    out: dict[str, float | None] = {"accuracy": metrics["accuracy"]}
    if dataset == "mme":
        out["mme_score"] = metrics["mme_total_score"]
    return out


def _mean(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize_run_file(path: Path, baseline_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path} is empty")
    meta_path = path.parent / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    dataset = rows[0]["dataset"]
    baseline_rows = {row["sample_id"]: row for row in read_jsonl(baseline_dir / path.name)} if (baseline_dir / path.name).exists() else {}
    paired = [baseline_rows[row["sample_id"]] for row in rows if row["sample_id"] in baseline_rows]

    summary: dict[str, Any] = {
        "run_id": meta.get("run_id", path.parent.name),
        "method": rows[0].get("method", meta.get("method")),
        "variant": rows[0].get("variant", meta.get("variant")),
        "level": rows[0].get("level", meta.get("level")),
        "source": path.stem,
        "dataset": dataset,
        "rows": len(rows),
        **{f"{key}": value for key, value in quality(dataset, rows).items()},
    }
    if len(paired) == len(rows):
        base_quality = quality(dataset, paired)
        summary.update({f"baseline_{key}": value for key, value in base_quality.items()})
        if summary["accuracy"] is not None and base_quality["accuracy"]:
            summary["relative_accuracy"] = summary["accuracy"] / base_quality["accuracy"]
        summary["agree_with_baseline"] = sum(
            row["prediction"].strip().lower() == baseline_rows[row["sample_id"]]["prediction"].strip().lower() for row in rows
        ) / len(rows)
        summary["baseline_input_tokens"] = _mean(row["input_tokens"] for row in paired)
    total_k = sum(row.get("K", 0) for row in rows)
    total_n = sum(row.get("N", 0) for row in rows)
    summary.update({
        "keep_ratio_realized": total_k / total_n if total_n else None,
        "K_mean": _mean(row.get("K") for row in rows),
        "N_mean": _mean(row.get("N") for row in rows),
        "input_tokens_mean": _mean(row.get("input_tokens") for row in rows),
        "latency_mean": _mean(row.get("latency_seconds") for row in rows),
        **{f"{key}_mean": _mean((row.get("timing") or {}).get(key) for row in rows) for key in TIMING_KEYS},
        "peak_vram_mb_max": max((row.get("peak_vram_mb") or 0.0) for row in rows),
    })
    if dataset == "mme":
        for low, high, name in MME_SIZE_BUCKETS:
            bucket = [row for row in rows if low < row.get("N", 0) <= high]
            correct = [score_prediction_record(row) for row in bucket]
            correct = [value for value in correct if value is not None]
            summary[f"mme_accuracy_{name}"] = sum(correct) / len(correct) if correct else None
            summary[f"mme_rows_{name}"] = len(bucket)
    return summary


def collect_summaries(run_dirs: Iterable[Path], baseline_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for run_dir in run_dirs:
        for path in sorted(run_dir.glob("*.jsonl")):
            if path.name == "errors.jsonl":
                continue
            summaries.append(summarize_run_file(path, baseline_dir))
    return summaries


def write_tables(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for summary in summaries:
        keys.extend(key for key in summary if key not in keys)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    columns = ["dataset", "method", "variant", "level", "rows", "keep_ratio_realized", "K_mean", "input_tokens_mean",
               "accuracy", "baseline_accuracy", "relative_accuracy", "agree_with_baseline", "mme_score",
               "baseline_mme_score", "latency_mean", "vit_mean", "selector_mean", "prefill_mean", "decode_mean"]
    fmt = lambda value: "—" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))  # noqa: E731
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for summary in sorted(summaries, key=lambda s: (s["dataset"], s["method"], s["variant"], -(s["keep_ratio_realized"] or 0))):
        lines.append("| " + " | ".join(fmt(summary.get(column)) for column in columns) + " |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summaries(summaries: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_dataset[summary["dataset"]].append(summary)

    for dataset, items in by_dataset.items():
        metric = "mme_score" if dataset == "mme" else "accuracy"
        fig, ax = plt.subplots(figsize=(7, 4.5))
        series: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            series[f"{item['method']}:{item['variant']}"].append(item)
        for label, points in sorted(series.items()):
            points = sorted(points, key=lambda p: p["keep_ratio_realized"] or 0)
            ax.plot([p["keep_ratio_realized"] for p in points], [p[metric] for p in points], marker="o", label=label)
        baselines = [item.get(f"baseline_{metric}") for item in items if item.get(f"baseline_{metric}") is not None]
        if baselines:
            ax.axhline(sum(baselines) / len(baselines), color="black", linestyle="--", linewidth=1, label="full token (same ids)")
        ax.set_xscale("log")
        ax.set_xlabel("realized keep ratio  ΣK / ΣN  (log scale)")
        ax.set_ylabel(metric)
        ax.set_title(f"{dataset}: quality vs. visual tokens kept")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        path = output_dir / f"quality_vs_keep_{dataset}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(items)), 4.5))
        ordered = sorted(items, key=lambda s: (s["method"], s["variant"], -(s["keep_ratio_realized"] or 0)))
        bottom = [0.0] * len(ordered)
        for key in TIMING_KEYS:
            values = [item.get(f"{key}_mean") or 0.0 for item in ordered]
            ax.bar(range(len(ordered)), values, bottom=bottom, label=key)
            bottom = [b + v for b, v in zip(bottom, values, strict=True)]
        ax.set_xticks(range(len(ordered)), [f"{i['variant']}\n{i['level']}" for i in ordered], fontsize=7, rotation=45)
        ax.set_ylabel("seconds per sample (mean)")
        ax.set_title(f"{dataset}: latency breakdown")
        ax.legend(fontsize=8)
        path = output_dir / f"latency_breakdown_{dataset}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def plot_token_usage(run_files: Iterable[Path], output_dir: Path) -> list[Path]:
    """Histogram of per-sample keep ratio K/N and of LLM input length, one figure per run file."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for path in run_files:
        rows = read_jsonl(path)
        if not rows or "K" not in rows[0]:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        axes[0].hist([row["K"] / row["N"] for row in rows], bins=30, color="tab:blue")
        axes[0].set_xlabel("K / N per sample")
        axes[0].set_ylabel("samples")
        axes[1].hist([row["input_tokens"] for row in rows], bins=30, color="tab:orange")
        axes[1].set_xlabel("LLM input tokens per sample (N_total')")
        fig.suptitle(f"{path.parent.name} / {path.stem}", fontsize=9)
        fig.tight_layout()
        target = output_dir / f"token_usage_{path.parent.name}_{path.stem}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=130)
        plt.close(fig)
        written.append(target)
    return written


def render_kept_tokens(image, grid_hw: tuple[int, int], keep_indices: Iterable[int], *, dim: float = 0.7):
    """Image resized to the merged grid (32 px per visual token); dropped tokens darkened, kept ones outlined."""

    from PIL import Image, ImageDraw

    height, width = grid_hw
    cell = 32
    canvas = image.convert("RGB").resize((width * cell, height * cell), Image.BICUBIC)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    kept = set(int(i) for i in keep_indices)
    for index in range(height * width):
        row, col = divmod(index, width)
        box = (col * cell, row * cell, (col + 1) * cell - 1, (row + 1) * cell - 1)
        if index in kept:
            draw.rectangle(box, outline=(0, 255, 0, 255), width=1)
        else:
            draw.rectangle(box, fill=(0, 0, 0, int(255 * dim)))
    return Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
