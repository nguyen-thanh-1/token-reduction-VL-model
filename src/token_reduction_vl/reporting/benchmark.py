"""Build a structured quality-and-efficiency report from prediction JSONL."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from token_reduction_vl.evaluation.metrics import score_prediction_records


REPORT_SCHEMA_VERSION = "1.1"
COLORS = {
    "gqa": "#4C78A8",
    "mmbench": "#F58518",
    "mme": "#54A24B",
}


@dataclass(frozen=True)
class ReportPaths:
    root: Path
    figures: Path
    tables: Path
    report: Path
    manifest: Path
    summary: Path

    @classmethod
    def from_root(cls, root: Path) -> ReportPaths:
        return cls(
            root=root,
            figures=root / "figures",
            tables=root / "tables",
            report=root / "report.md",
            manifest=root / "manifest.json",
            summary=root / "summary.json",
        )


@dataclass
class BenchmarkRun:
    key: str
    label: str
    path: Path
    records: list[dict[str, Any]]
    metrics: dict[str, Any]
    performance: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            records.append(value)
    if not records:
        raise ValueError(f"Prediction file is empty: {path}")
    return records


def _numeric_values(records: Iterable[Mapping[str, Any]], field: str) -> list[float]:
    return [
        float(record[field])
        for record in records
        if record.get(field) is not None
    ]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summarize_performance(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize runtime fields while checking basic output integrity."""

    sample_ids = [str(record.get("sample_id", "")) for record in records]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("At least one prediction record is missing sample_id.")
    duplicate_count = len(sample_ids) - len(set(sample_ids))
    latencies = _numeric_values(records, "latency_seconds")
    input_tokens = _numeric_values(records, "input_tokens")
    output_tokens = _numeric_values(records, "output_tokens")
    peak_vram = _numeric_values(records, "peak_vram_mb")
    if not latencies or not input_tokens or not output_tokens:
        raise ValueError("Predictions are missing runtime/token measurements.")

    return {
        "rows": len(records),
        "unique_sample_ids": len(set(sample_ids)),
        "duplicate_sample_ids": duplicate_count,
        "empty_predictions": sum(
            not str(record.get("prediction", "")).strip() for record in records
        ),
        "mean_latency_seconds": mean(latencies),
        "median_latency_seconds": median(latencies),
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "max_latency_seconds": max(latencies),
        "total_generation_minutes": sum(latencies) / 60.0,
        "mean_input_tokens": mean(input_tokens),
        "p95_input_tokens": _percentile(input_tokens, 0.95),
        "mean_output_tokens": mean(output_tokens),
        "max_output_tokens": max(output_tokens),
        "max_peak_vram_mb": max(peak_vram) if peak_vram else None,
    }


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_benchmark_runs(
    predictions: Mapping[str, Any], *, project_root: Path
) -> list[BenchmarkRun]:
    runs: list[BenchmarkRun] = []
    for key in ("gqa", "mmbench", "mme"):
        if key not in predictions:
            raise KeyError(f"Missing prediction configuration for {key!r}.")
        source = predictions[key]
        path = _resolve(project_root, source["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file does not exist: {path}")
        records = _read_jsonl(path)
        metrics = score_prediction_records(records)
        if metrics["dataset"] != key:
            raise ValueError(
                f"Configured key {key!r} contains dataset {metrics['dataset']!r}."
            )
        runs.append(
            BenchmarkRun(
                key=key,
                label=str(source.get("label", key)),
                path=path,
                records=records,
                metrics=metrics,
                performance=summarize_performance(records),
            )
        )
    return runs


def _prepare_output(paths: ReportPaths, *, overwrite: bool) -> None:
    existing_files = list(paths.root.rglob("*")) if paths.root.exists() else []
    if any(path.is_file() for path in existing_files) and not overwrite:
        raise FileExistsError(
            f"Report output already contains files: {paths.root}. "
            "Use --overwrite to regenerate this same report."
        )
    paths.figures.mkdir(parents=True, exist_ok=True)
    paths.tables.mkdir(parents=True, exist_ok=True)


def _save_figure(figure: plt.Figure, path: Path, *, dpi: int) -> None:
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _annotate_bars(axis: plt.Axes, values: list[float], *, suffix: str = "") -> None:
    for patch, value in zip(axis.patches, values, strict=True):
        axis.annotate(
            f"{value:.1f}{suffix}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )


def _plot_quality(runs: list[BenchmarkRun], output: Path, *, dpi: int) -> None:
    labels = [run.label for run in runs]
    values = [100.0 * float(run.metrics["accuracy"]) for run in runs]
    colors = [COLORS[run.key] for run in runs]
    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.bar(labels, values, color=colors, width=0.62)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Full-token BF16 baseline: answer quality")
    axis.grid(axis="y", alpha=0.25)
    _annotate_bars(axis, values, suffix="%")
    axis.text(
        0.5,
        -0.18,
        "Dataset-specific protocols; these percentages should not be averaged into one score.",
        transform=axis.transAxes,
        ha="center",
        fontsize=9,
        color="#555555",
    )
    _save_figure(figure, output, dpi=dpi)


def _plot_efficiency(runs: list[BenchmarkRun], output: Path, *, dpi: int) -> None:
    labels = [run.key.upper() for run in runs]
    colors = [COLORS[run.key] for run in runs]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    mean_latency = [run.performance["mean_latency_seconds"] for run in runs]
    p95_latency = [run.performance["p95_latency_seconds"] for run in runs]
    positions = list(range(len(runs)))
    axes[0].bar(
        [position - 0.18 for position in positions],
        mean_latency,
        width=0.36,
        label="Mean",
        color=colors,
    )
    axes[0].bar(
        [position + 0.18 for position in positions],
        p95_latency,
        width=0.36,
        label="P95",
        color=colors,
        alpha=0.45,
        hatch="//",
    )
    axes[0].set_xticks(positions, labels)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Seconds (log scale)")
    axes[0].set_title("Generation latency")
    axes[0].legend(frameon=False)

    input_tokens = [run.performance["mean_input_tokens"] for run in runs]
    axes[1].bar(labels, input_tokens, color=colors)
    axes[1].set_ylabel("Tokens per sample")
    axes[1].set_title("Mean input tokens")
    _annotate_bars(axes[1], input_tokens)

    peak_vram = [run.performance["max_peak_vram_mb"] / 1024.0 for run in runs]
    axes[2].bar(labels, peak_vram, color=colors)
    axes[2].set_ylabel("GiB")
    axes[2].set_title("Maximum observed VRAM")
    _annotate_bars(axes[2], peak_vram)

    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Full-token BF16 baseline: efficiency", fontsize=14)
    figure.tight_layout()
    _save_figure(figure, output, dpi=dpi)


def _plot_latency_distribution(
    runs: list[BenchmarkRun], output: Path, *, dpi: int
) -> None:
    values = [_numeric_values(run.records, "latency_seconds") for run in runs]
    labels = [run.key.upper() for run in runs]
    figure, axis = plt.subplots(figsize=(9, 5.4))
    boxes = axis.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, run in zip(boxes["boxes"], runs, strict=True):
        patch.set_facecolor(COLORS[run.key])
        patch.set_alpha(0.75)
    axis.set_yscale("log")
    axis.set_ylabel("Latency per sample (seconds, log scale)")
    axis.set_title("Generation latency distribution (outliers hidden)")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output, dpi=dpi)


def _sample_records(records: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if len(records) <= maximum:
        return records
    step = len(records) / maximum
    return [records[int(index * step)] for index in range(maximum)]


def _plot_tokens_vs_latency(
    runs: list[BenchmarkRun],
    output: Path,
    *,
    dpi: int,
    max_points_per_dataset: int,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.8))
    for run in runs:
        sampled = _sample_records(run.records, max_points_per_dataset)
        axis.scatter(
            _numeric_values(sampled, "input_tokens"),
            _numeric_values(sampled, "latency_seconds"),
            s=11,
            alpha=0.28,
            color=COLORS[run.key],
            label=run.key.upper(),
            edgecolors="none",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Input tokens (log scale)")
    axis.set_ylabel("Generation latency (seconds, log scale)")
    axis.set_title("Input-token count versus generation latency")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _save_figure(figure, output, dpi=dpi)


def _plot_mme_categories(run: BenchmarkRun, output: Path, *, dpi: int) -> None:
    categories = run.metrics.get("categories", {})
    ordered = sorted(categories.items(), key=lambda item: item[1]["score"])
    labels = [name.replace("_", " ") for name, _ in ordered]
    accuracy = [values["accuracy_percent"] for _, values in ordered]
    accuracy_plus = [values["accuracy_plus_percent"] for _, values in ordered]
    scores = [values["score"] for _, values in ordered]
    positions = list(range(len(labels)))

    figure, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    axes[0].barh(
        [position - 0.2 for position in positions],
        accuracy,
        height=0.4,
        label="Accuracy",
        color="#4C78A8",
    )
    axes[0].barh(
        [position + 0.2 for position in positions],
        accuracy_plus,
        height=0.4,
        label="Accuracy+",
        color="#E45756",
    )
    axes[0].set_yticks(positions, labels)
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Percent")
    axes[0].set_title("Question and pair accuracy")
    axes[0].legend(frameon=False)

    axes[1].barh(positions, scores, color="#54A24B")
    axes[1].set_xlim(0, 200)
    axes[1].set_xlabel("MME category score")
    axes[1].set_title("Accuracy + Accuracy+")
    for position, score in zip(positions, scores, strict=True):
        axes[1].text(score + 2, position, f"{score:.1f}", va="center", fontsize=8)

    for axis in axes:
        axis.grid(axis="x", alpha=0.22)
    figure.suptitle("MME performance by category", fontsize=14)
    figure.tight_layout()
    _save_figure(figure, output, dpi=dpi)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _benchmark_rows(runs: list[BenchmarkRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.append(
            {
                "dataset": run.key,
                "split_label": run.label,
                "rows": run.performance["rows"],
                "accuracy_percent": 100.0 * run.metrics["accuracy"],
                "mme_total_score": run.metrics.get("mme_total_score"),
                "mean_latency_seconds": run.performance["mean_latency_seconds"],
                "median_latency_seconds": run.performance["median_latency_seconds"],
                "p95_latency_seconds": run.performance["p95_latency_seconds"],
                "max_latency_seconds": run.performance["max_latency_seconds"],
                "mean_input_tokens": run.performance["mean_input_tokens"],
                "p95_input_tokens": run.performance["p95_input_tokens"],
                "mean_output_tokens": run.performance["mean_output_tokens"],
                "max_peak_vram_mb": run.performance["max_peak_vram_mb"],
                "empty_predictions": run.performance["empty_predictions"],
                "duplicate_sample_ids": run.performance["duplicate_sample_ids"],
            }
        )
    return rows


def _mme_rows(run: BenchmarkRun) -> list[dict[str, Any]]:
    return [
        {"category": category, **values}
        for category, values in sorted(run.metrics.get("categories", {}).items())
    ]


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _portable(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _format_mme_score(run: BenchmarkRun) -> str:
    score = run.metrics.get("mme_total_score")
    return "—" if score is None else f"{float(score):.2f}"


def _render_report(
    *,
    title: str,
    runs: list[BenchmarkRun],
    mme: BenchmarkRun,
    model: Mapping[str, Any],
    max_scatter_points: int,
) -> str:
    quality_rows = "\n".join(
        f"| {run.label} | {run.metrics['labelled_rows']:,} | "
        f"{100.0 * run.metrics['accuracy']:.2f}% | "
        f"{_format_mme_score(run)} |"
        for run in runs
    )
    runtime_rows = "\n".join(
        f"| {run.label} | {run.performance['mean_latency_seconds']:.4f} | "
        f"{run.performance['median_latency_seconds']:.4f} | "
        f"{run.performance['p95_latency_seconds']:.4f} | "
        f"{run.performance['max_latency_seconds']:.4f} | "
        f"{run.performance['total_generation_minutes']:.2f} | "
        f"{run.performance['max_peak_vram_mb'] / 1024.0:.2f} |"
        for run in runs
    )
    token_rows = "\n".join(
        f"| {run.label} | "
        f"{run.performance['mean_input_tokens']:.1f} | "
        f"{run.performance['p95_input_tokens']:.1f} | "
        f"{run.performance['mean_output_tokens']:.1f} | "
        f"{run.performance['max_output_tokens']:.0f} |"
        for run in runs
    )
    integrity_rows = "\n".join(
        f"| {run.label} | {run.performance['rows']:,} | "
        f"{run.performance['unique_sample_ids']:,} | "
        f"{run.performance['duplicate_sample_ids']:,} | "
        f"{run.performance['empty_predictions']:,} |"
        for run in runs
    )
    categories = mme.metrics["categories"]
    category_count = len(categories)
    maximum_mme_score = 200 * category_count
    best_name, best = max(categories.items(), key=lambda item: item[1]["score"])
    worst_name, worst = min(categories.items(), key=lambda item: item[1]["score"])
    return f"""# {title}

## Run identity

- Model: `{model['model_id']}`
- Precision: `{model['precision']}`
- Pruning method: `{model['pruning_method']}`
- Report schema: `{REPORT_SCHEMA_VERSION}`

## How to interpret this report

Quality and efficiency are deliberately reported as separate axes. Higher
accuracy and MME scores are better; lower latency, token count, and VRAM are
better. A useful token-reduction method should reduce computation while keeping
the quality loss small. The three benchmark accuracies must not be averaged
because their tasks and scoring protocols differ.

## Metric definitions and rationale

### Answer-quality metrics

| Metric | Definition used in this project | Why it is reported |
| --- | --- | --- |
| Labelled rows | Samples with an available reference answer or answer label. Unlabelled rows are excluded from the accuracy denominator. | Makes the evaluated sample count explicit and prevents missing labels from silently lowering accuracy. |
| GQA accuracy | Percentage of labelled questions whose prediction exactly matches any reference after trimming, case-folding, and collapsing repeated whitespace. | Directly measures open-ended visual question answering quality on the balanced GQA split. |
| MMBench accuracy | Percentage of labelled questions with the correct A–E option. The parser accepts an option label or an exact match to the option text. | Measures multiple-choice multimodal reasoning while handling the two common answer formats. |
| MME question accuracy | Percentage of individual yes/no questions answered correctly. The first `yes` or `no` token in the prediction is compared with the reference. | Gives an intuitive question-level percentage, but it is supplementary to the official-style MME score. |
| MME Accuracy+ | Within one category, the percentage of complete image pairs for which both paired yes/no questions are correct. | Penalizes inconsistent perception: one correct answer from a positive/negative pair is not sufficient. |
| MME category score | `question accuracy (%) + Accuracy+ (%)`; range 0–200 for each category. | Preserves both per-question correctness and pair consistency. |
| MME total score | Sum of all category scores. This report contains {category_count} categories, so the observed scale is 0–{maximum_mme_score}. | This is the primary aggregate used to compare MME runs; it is not a percentage and should not be compared numerically with GQA/MMBench accuracy. |

### Efficiency metrics

| Metric | Definition used in this project | Why it is reported |
| --- | --- | --- |
| Generation latency | Wall-clock seconds around `model.generate()` for one sample. Image preprocessing, prompt construction, decoding text, file I/O, and metric computation are excluded. | Isolates model-generation cost, which visual-token pruning is intended to reduce. |
| Mean latency | Arithmetic mean of per-sample generation latency. | Summarizes overall runtime, but can be pulled upward by slow samples. |
| Median latency | Middle per-sample latency. | Represents a typical sample and is robust to a long latency tail. |
| P95 latency | Deterministic sorted-index estimate of the 95th percentile; approximately 95% of samples are no slower than this value. | Captures tail latency that the mean can hide, especially for high-resolution images with many visual tokens. |
| Maximum latency | Slowest recorded sample. | Exposes worst observed behavior and helps identify samples for diagnosis. |
| Input tokens | Length of the processor-produced input sequence, including text tokens and visual placeholder positions replaced by visual embeddings. | Serves as a direct workload proxy for the language decoder. |
| Output tokens | Number of tokens generated after the input prompt. | Separates response-length effects from input visual-token effects. |
| Peak VRAM | Maximum CUDA memory allocated during generation after resetting peak-memory statistics; includes resident model memory plus generation-time tensors and KV cache. | Shows whether a method improves memory feasibility, not only speed. |

The current full-token baseline retains 100% of the visual tokens emitted by
Qwen3-VL's built-in 2×2 spatial patch merger. Future pruning reports should add
the retained visual-token count and ratio while keeping these definitions
unchanged.

## Quality

| Benchmark | Labelled rows | Accuracy | MME score |
| --- | ---: | ---: | ---: |
{quality_rows}

![Quality overview](figures/01_quality_accuracy.png)

**Figure 1 — Quality accuracy.** Each bar is the dataset-specific percentage
defined above. Use it to compare the same benchmark across model or pruning
runs. Do not average the three bars or compare the MME percentage with its
separate total score.

## Efficiency

### Runtime and memory

| Benchmark | Mean latency (s) | Median (s) | P95 (s) | Maximum (s) | Total generation (min) | Peak VRAM (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{runtime_rows}

### Token counts

| Benchmark | Mean input tokens | P95 input tokens | Mean output tokens | Maximum output tokens |
| --- | ---: | ---: | ---: | ---: |
{token_rows}

![Efficiency overview](figures/02_efficiency_overview.png)

**Figure 2 — Efficiency overview.** The first panel compares mean and P95
generation latency on a logarithmic scale, the second compares mean input
tokens, and the third shows maximum observed VRAM. Mean describes aggregate
cost; P95 and maximum VRAM expose difficult high-cost samples.

![Latency distribution](figures/03_latency_distribution.png)

**Figure 3 — Latency distribution.** Box centers are medians, boxes span the
25th–75th percentiles, and whiskers use the standard 1.5×IQR rule. Outlier
markers are hidden to keep the three distributions readable, but those samples
remain included in every table statistic. The vertical axis is logarithmic.

![Tokens versus latency](figures/04_tokens_vs_latency.png)

**Figure 4 — Input tokens versus latency.** Each point is one sample; both axes
are logarithmic. An upward trend indicates that longer multimodal sequences
cost more generation time and therefore offer an opportunity for visual-token
reduction. For readability, at most {max_scatter_points:,} deterministically
spaced records per dataset are plotted; table metrics always use every record.

## MME category analysis

- Highest category score: `{best_name}` ({best['score']:.2f}).
- Lowest category score: `{worst_name}` ({worst['score']:.2f}).
- Total MME score: {mme.metrics['mme_total_score']:.2f}.

![MME categories](figures/05_mme_categories.png)

**Figure 5 — MME categories.** The left panel separates question Accuracy from
pair Accuracy+. The right panel shows their sum, sorted from weakest to
strongest category. This reveals whether a pruning method damages particular
capabilities even when the total score appears stable.

## Output integrity

| Benchmark | Rows | Unique sample IDs | Duplicate IDs | Empty predictions |
| --- | ---: | ---: | ---: | ---: |
{integrity_rows}

Duplicate IDs or empty predictions should be zero before results are treated
as a valid benchmark. These checks verify output completeness; they do not
replace dataset-version and prompt-configuration provenance in `manifest.json`.

## Artifact layout

- `summary.json`: machine-readable metrics and efficiency statistics.
- `tables/benchmark_summary.csv`: one row per benchmark.
- `tables/mme_categories.csv`: MME category-level values.
- `manifest.json`: report identity, input provenance, and generated files.
- `figures/`: deterministic numbered figures in reading order.
"""


def build_benchmark_report(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build all report artifacts and return the written manifest."""

    report_config = config["report"]
    visualization = config.get("visualization", {})
    root = output_root or _resolve(project_root, report_config["output_dir"])
    paths = ReportPaths.from_root(root)
    _prepare_output(paths, overwrite=overwrite)
    runs = load_benchmark_runs(config["predictions"], project_root=project_root)
    mme = next(run for run in runs if run.key == "mme")
    dpi = int(visualization.get("dpi", 180))
    max_points = int(visualization.get("max_scatter_points_per_dataset", 5000))

    figure_names = (
        "01_quality_accuracy.png",
        "02_efficiency_overview.png",
        "03_latency_distribution.png",
        "04_tokens_vs_latency.png",
        "05_mme_categories.png",
    )
    _plot_quality(runs, paths.figures / figure_names[0], dpi=dpi)
    _plot_efficiency(runs, paths.figures / figure_names[1], dpi=dpi)
    _plot_latency_distribution(runs, paths.figures / figure_names[2], dpi=dpi)
    _plot_tokens_vs_latency(
        runs,
        paths.figures / figure_names[3],
        dpi=dpi,
        max_points_per_dataset=max_points,
    )
    _plot_mme_categories(mme, paths.figures / figure_names[4], dpi=dpi)

    benchmark_rows = _benchmark_rows(runs)
    benchmark_fields = list(benchmark_rows[0])
    _write_csv(paths.tables / "benchmark_summary.csv", benchmark_rows, benchmark_fields)
    mme_rows = _mme_rows(mme)
    _write_csv(
        paths.tables / "mme_categories.csv",
        mme_rows,
        ["category", "questions", "pairs", "accuracy_percent", "accuracy_plus_percent", "score"],
    )
    summary = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "benchmarks": {
            run.key: {
                "label": run.label,
                "metrics": run.metrics,
                "performance": run.performance,
            }
            for run in runs
        },
    }
    _atomic_text(paths.summary, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    _atomic_text(
        paths.report,
        _render_report(
            title=str(report_config["title"]),
            runs=runs,
            mme=mme,
            model=report_config,
            max_scatter_points=max_points,
        ),
    )

    generated_files = [
        "report.md",
        "summary.json",
        "tables/benchmark_summary.csv",
        "tables/mme_categories.csv",
        *(f"figures/{name}" for name in figure_names),
    ]
    manifest = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_id": report_config["id"],
        "generated_at": datetime.now(UTC).isoformat(),
        "model_id": report_config["model_id"],
        "precision": report_config["precision"],
        "pruning_method": report_config["pruning_method"],
        "inputs": {
            run.key: {
                "path": _portable(run.path, project_root),
                "size_bytes": run.path.stat().st_size,
                "rows": len(run.records),
                "modified_at": datetime.fromtimestamp(
                    run.path.stat().st_mtime, tz=UTC
                ).isoformat(),
            }
            for run in runs
        },
        "generated_files": generated_files,
    }
    _atomic_text(paths.manifest, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
