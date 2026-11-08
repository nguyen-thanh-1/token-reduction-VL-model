"""Figures for the DeSTiL paper, from the run files already on disk (no model, no GPU).

Writes PNGs to --output-dir (default docs/paper/figures):
    fig_budget_curves.png      quality vs kept share, GQA / MMB / MME perception (final runs)
    fig_mme_subtasks.png       MME score per subtask, full vs DeSTiL L3 / L4
    fig_timing.png             median ViT / selector / prefill / decode per sample (timing2 runs, shuffled order)
    fig_region.png             compactness and question dependence of kept sets (tune2k)
    fig_ablation.png           development ablation at L1-L4 (tune2k)
    fig_selection.png          kept tokens of four selectors on GQA testdev images: (a) DeSTiL right,
                               DivPrune and FastV-style wrong; (b) DeSTiL wrong, DivPrune right
    fig_focus_questions.png    DeSTiL-fc focus regions for different questions on one image

Example:
    uv run python scripts/pp3/paper_figures.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from token_reduction_vl.evaluation.metrics import score_prediction_record, score_prediction_records  # noqa: E402
from token_reduction_vl.reporting.pruning_report import read_jsonl  # noqa: E402
from token_reduction_vl.reporting.region_metrics import summarize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_report import MME_COGNITION, mme_scores, run_path  # noqa: E402

# Categorical slots in fixed order (dataviz reference palette, light mode).
METHODS = (
    ("3d_attn50k", "DeSTiL (ours)", "#2a78d6", "o"),
    ("divprune", "DivPrune", "#eb6834", "s"),
    ("fastv", "FastV-style", "#1baf7a", "^"),
    ("random", "Random", "#eda100", "D"),
    ("cdpruner", "CDPruner", "#e87ba4", "v"),
)
# Figures are drawn at their printed size (IEEE two-column: one column 3.5 in, full width 7.16 in),
# so the font sizes below are the sizes on paper.
COLUMN_IN = 3.5
TEXT_IN = 7.16
FULL_COLOR = "#52514e"
MUTED = "#8c8b86"
LEVELS = ("L1", "L2", "L3", "L4")
SOURCES = {"gqa": "gqa_testdev_balanced", "mmb": "mmb_validation", "mme": "mme_test"}
CANONICAL = {
    "gqa_testdev_balanced": "data/processed/v1/gqa/testdev.jsonl",
    "gqa_val_balanced": "data/processed/v1/gqa/val.jsonl",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": "#0b0b0b", "xtick.color": "#3a3936", "ytick.color": "#3a3936",
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "legend.frameon": False,
    })


def load(prefix: str, name: str, dataset: str, source: str) -> list[dict]:
    return read_jsonl(run_path(prefix, name, dataset, source))


def accuracy(rows: list[dict]) -> float:
    return 100 * sum(bool(score_prediction_record(r)) for r in rows) / len(rows)


def kept_share(rows: list[dict]) -> float:
    return sum(r["K"] for r in rows) / sum(r["N"] for r in rows)


def fig_budget_curves(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, 2.45))
    panels = (("gqa", "GQA testdev: Δ accuracy vs. full (pts)"),
              ("mmb", "MMBench dev: Δ accuracy vs. full (pts)"),
              ("mme", "MME test: perception score"))
    for ax, (dataset, title) in zip(axes, panels):
        source = SOURCES[dataset]
        full = load("final", "full", dataset, source)
        full_value = mme_scores(full)[0] if dataset == "mme" else accuracy(full)
        for name, label, color, marker in METHODS:
            xs, ys = [], []
            for level in LEVELS:
                rows = load("final", f"{name}_{level}", dataset, source)
                value = mme_scores(rows)[0] if dataset == "mme" else accuracy(rows)
                xs.append(100 * kept_share(rows))
                ys.append(value if dataset == "mme" else value - full_value)
            ax.plot(xs, ys, color=color, marker=marker, markersize=4, linewidth=1.5, label=label)
        ref = full_value if dataset == "mme" else 0.0
        ax.axhline(ref, color=FULL_COLOR, linestyle="--", linewidth=1.0, label="Full (100% tokens)")
        if dataset != "mme":
            ax.axhspan(-1.0, 0.0, color="#e6e5e1", alpha=0.6, linewidth=0, label="1-pt non-inferiority margin")
        ax.set_title(title, loc="left")
        ax.set_xticks([5.7, 11.1, 22.2, 33.3], ["5.7\nL1", "11\nL2", "22\nL3", "33\nL4"])
    axes[1].set_xlabel("Visual tokens kept (%)")
    fig.tight_layout(w_pad=1.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out / "fig_budget_curves.png")
    plt.close(fig)


def subtask_scores(rows: list[dict]) -> dict[str, float]:
    return {name: c["score"] for name, c in score_prediction_records(rows)["categories"].items()}


def fig_mme_subtasks(out: Path) -> None:
    series = (("full", "Full", FULL_COLOR), ("3d_attn50k_L3", "DeSTiL L3 (22%)", "#2a78d6"),
              ("3d_attn50k_L4", "DeSTiL L4 (33%)", "#eb6834"))
    scores = {name: subtask_scores(load("final", name, "mme", "mme_test")) for name, _, _ in series}
    tasks = sorted(scores["full"], key=lambda t: (t in MME_COGNITION, t))
    y = np.arange(len(tasks))
    height = 0.26
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 4.4))
    for i, (name, label, color) in enumerate(series):
        ax.barh(y + (i - 1) * height, [scores[name][t] for t in tasks], height=height - 0.03, color=color, label=label)
    ax.set_yticks(y, [f"{t} *" if t in MME_COGNITION else t for t in tasks])
    ax.invert_yaxis()
    ax.set_xlim(0, 200)
    ax.set_xlabel("MME score per subtask (max 200)")
    ax.set_title("MME by subtask (* = cognition)", loc="left")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.42, -0.1), handlelength=1.2, columnspacing=0.8)
    fig.savefig(out / "fig_mme_subtasks.png")
    plt.close(fig)


def fig_timing(out: Path) -> None:
    # FastV-style is left out: its extra two-layer pass is not inside the timed selector.
    runs = (("full", "Full"), ("random_L3", "Random L3"), ("divprune_L3", "DivPrune L3"), ("cdpruner_L3", "CDPruner L3"),
            ("3d_attn50k_L1", "DeSTiL L1"), ("3d_attn50k_L3", "DeSTiL L3"), ("3d_attn50k_L4", "DeSTiL L4"))
    parts = (("vit", "ViT (shared)", "#c3c2b7"), ("selector", "Token selection", "#6250d6"),
             ("prefill", "LLM prefill", "#2a78d6"), ("decode", "Decode", "#1baf7a"))
    # Stacked vertically so the figure fits one column of a two-column paper.
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 4.6))
    for ax, (dataset, source, title) in zip(axes, (("gqa", "gqa_val_balanced", "GQA (small images, ~220 tokens to the LLM)"),
                                                   ("mme", "mme_test", "MME (large images, ~900 tokens to the LLM)"))):
        y = np.arange(len(runs))
        left = np.zeros(len(runs))
        for key, label, color in parts:
            values = np.array([1000 * np.median([r["timing"].get(key, 0.0) for r in load("timing2", n, dataset, source)])
                               for n, _ in runs])
            ax.barh(y, values, left=left, color=color, label=label, height=0.62, edgecolor="white", linewidth=1)
            left += values
        for yi, total in zip(y, left):
            ax.text(total + 0.01 * left.max(), yi, f"{total:.0f} ms", va="center", fontsize=7.5, color="#0b0b0b")
        ax.set_yticks(y, [label for _, label in runs])
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.set_xlim(0, left.max() * 1.25)
    axes[1].set_xlabel("ms per question (median per stage, 300 questions)")
    fig.tight_layout()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out / "fig_timing.png")
    plt.close(fig)


def focus_rows(rows: list[dict]) -> list[dict]:
    return [{**r, "keep_indices": r["pruner_metadata"]["focus_indices"], "K": len(r["pruner_metadata"]["focus_indices"])}
            for r in rows if (r.get("pruner_metadata") or {}).get("focus_indices")]


def fig_region(out: Path) -> None:
    def tune(name: str) -> list[dict]:
        return load("tune2k", name, "gqa", "gqa_val_balanced")

    bars = (("Random L2 (11%)", summarize(tune("random_L2")), "#eda100"),
            ("Random L3 (22%)", summarize(tune("random_L3")), "#eda100"),
            ("DeSTiL L3 (22%)", summarize(tune("3d_attn50k_L3")), "#2a78d6"),
            ("DeSTiL L4 (33%)", summarize(tune("3d_attn50k_L4")), "#2a78d6"),
            ("DeSTiL L5 (50%)", summarize(tune("3d_attn50k_L5")), "#2a78d6"),
            ("DeSTiL-fc L4, all (33%)", summarize(tune("fcd25_L4")), "#e87ba4"),
            ("DeSTiL-fc L5, all (50%)", summarize(tune("fcd25_L5")), "#e87ba4"),
            ("DeSTiL-fc L4, focus (8.5%)", summarize(focus_rows(tune("fcd25_L4"))), "#6250d6"),
            ("DeSTiL-fc L5, focus (12.5%)", summarize(focus_rows(tune("fcd25_L5"))), "#6250d6"))
    # Side by side at full text width, sharing the row labels; printed under the focus examples.
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.35), sharey=True)
    y = np.arange(len(bars))
    for ax, (key, title) in zip(axes, (("components_per_10", "Regions per 10 kept tokens (lower = more compact)"),
                                      ("question_overlap", "Overlap between questions on one image\n(lower = more question-specific)"))):
        values = [s[key] for _, s, _ in bars]
        ax.barh(y, values, color=[c for *_, c in bars], height=0.6)
        for yi, v in zip(y, values):
            ax.text(v + 0.01 * max(values), yi, f"{v:.2f}", va="center", fontsize=7.5)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, max(values) * 1.22)
    axes[0].set_title("Regions per 10 kept tokens\n(lower = more compact)", loc="left")
    axes[0].set_yticks(y, [label for label, *_ in bars])
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "fig_region.png")
    plt.close(fig)


def fig_ablation(out: Path) -> None:
    # Development split tune2k (n = 2000, full = 68.90); docs/paper/pp3_draft.md §5.3.
    steps = (("Random (with fold)", (-20.10, -13.05, -6.95, -4.00), "#eda100"),
             ("CDPruner (DPP, mean cosine)", (-16.95, -9.65, -5.00, -2.75), "#e87ba4"),
             ("DSFold-D (hand-crafted score)", (-11.05, -6.05, -3.25, -0.95), "#008300"),
             ("H_MLP, gradient labels", (-3.15, -1.60, -0.50, -0.40), "#6250d6"),
             ("DeSTiL: H_TG, attention teacher", (-4.35, -1.20, -0.20, -0.20), "#2a78d6"))
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.5))
    x = np.arange(len(LEVELS))
    width = 0.16
    for i, (label, values, color) in enumerate(steps):
        ax.bar(x + (i - 2) * width, values, width=width - 0.02, color=color, label=label)
    ax.axhline(0, color=FULL_COLOR, linewidth=0.8)
    ax.axhspan(-1.0, 0.0, color="#e6e5e1", alpha=0.6, linewidth=0, zorder=0)
    ax.set_xticks(x, ["L1 (5.7%)", "L2 (11%)", "L3 (22%)", "L4 (33%)"])
    ax.set_ylabel("Δ accuracy vs. full (pts)")
    ax.set_title("Ablation on tune2k (GQA val, 2,000 questions)", loc="left")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.45, -0.13), handlelength=1.2, columnspacing=0.8)
    fig.savefig(out / "fig_ablation.png")
    plt.close(fig)


def canonical_images(source: str) -> dict[str, str]:
    import json

    with (PROJECT_ROOT / CANONICAL[source]).open(encoding="utf-8") as handle:
        return {row["sample_id"]: row["image"] for row in map(json.loads, handle)}


def overlay(ax, image: Image.Image, grid_hw: list[int], keep: set[int], focus: set[int] | None = None) -> None:
    """Dim the dropped tokens; outline focus tokens (if any) in red."""

    rows, cols = grid_hw
    width, height = image.size
    base = np.asarray(image.convert("RGB")).astype(float) / 255.0
    mask = np.zeros((rows, cols))
    for index in keep:
        mask[index // cols, index % cols] = 1.0
    big = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((width, height), Image.NEAREST)) / 255.0
    shaded = base * (0.25 + 0.75 * big[..., None])
    ax.imshow(shaded)
    if focus:
        cw, ch = width / cols, height / rows
        for index in focus:
            r, c = divmod(index, cols)
            ax.add_patch(plt.Rectangle((c * cw, r * ch), cw, ch, fill=False, edgecolor="#e34948", linewidth=1.4))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def fig_selection(out: Path, n_right: int = 2, n_wrong: int = 2) -> None:
    """One figure, two groups of GQA testdev questions (sorted by id, one per image, not yes/no,
    120 <= N <= 300) where full is right: (a) the first ``n_right`` where DeSTiL is right and DivPrune and
    FastV-style wrong; (b) the first ``n_wrong`` where DivPrune is right and DeSTiL wrong."""

    import textwrap

    source = SOURCES["gqa"]
    names = ("full", "3d_attn50k_L3", "divprune_L3", "fastv_L3", "random_L3")
    runs = {n: {r["sample_id"]: r for r in load("final", n, "gqa", source)} for n in names}
    images = canonical_images(source)

    def pick(wrong: bool, count: int, seen: set[str]) -> list[dict]:
        chosen = []
        for sample_id in sorted(runs["3d_attn50k_L3"]):
            rows = {n: runs[n].get(sample_id) for n in names}
            if any(r is None for r in rows.values()) or rows["full"]["image_id"] in seen:
                continue
            ok = {n: bool(score_prediction_record(r)) for n, r in rows.items()}
            wanted = (ok["divprune_L3"] and not ok["3d_attn50k_L3"]) if wrong else \
                (ok["3d_attn50k_L3"] and not ok["divprune_L3"] and not ok["fastv_L3"])
            if ok["full"] and wanted \
                    and str(rows["full"]["references"][0]).lower() not in ("yes", "no") and 120 <= rows["full"]["N"] <= 300:
                chosen.append(rows)
                seen.add(rows["full"]["image_id"])
            if len(chosen) == count:
                break
        return chosen

    groups = [("a", r) for r in pick(False, n_right, set())] + [("b", r) for r in pick(True, n_wrong, set())]
    titles = ("Original image (full)", "DeSTiL L3", "DivPrune L3", "FastV-style L3", "Random L3")
    fig, axes = plt.subplots(len(groups), len(names), figsize=(TEXT_IN, 1.62 * len(groups)))
    for row_axes, (tag, rows) in zip(np.atleast_2d(axes), groups):
        image = Image.open(PROJECT_ROOT / images[rows["full"]["sample_id"]])
        question = rows["full"]["question"].split("\n")[0]
        for ax, name, title in zip(row_axes, names, titles):
            r = rows[name]
            if name == "full":
                ax.imshow(image.convert("RGB"))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                ax.set_title(f"({tag}) " + textwrap.fill(question, 26) + f"\nanswer: {r['references'][0]}",
                             fontsize=7, loc="left")
                continue
            overlay(ax, image, r["grid_hw"], set(r["keep_indices"]))
            mark = "correct" if score_prediction_record(r) else "wrong"
            ax.set_title(f"{title}\n\"{r['prediction']}\" ({mark})", fontsize=7, loc="left")
    fig.tight_layout(h_pad=0.5, w_pad=0.4)
    fig.savefig(out / "fig_selection.png")
    plt.close(fig)


def fig_focus_questions(out: Path) -> None:
    rows = load("tune2k", "fcd25_L5", "gqa", "gqa_val_balanced")
    by_image: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if (r.get("pruner_metadata") or {}).get("focus_indices"):
            by_image[r["image_id"]].append(r)
    # The image whose questions have the most different focus sets (lowest mean Jaccard), 3 questions.
    from itertools import combinations

    from token_reduction_vl.reporting.region_metrics import jaccard

    def spread(group: list[dict]) -> float:
        sets = [set(r["pruner_metadata"]["focus_indices"]) for r in group]
        return float(np.mean([jaccard(a, b) for a, b in combinations(sets, 2)]))

    candidates = [g[:3] for g in by_image.values() if len(g) >= 3 and all(score_prediction_record(r) for r in g[:3])]
    group = min(candidates, key=spread)
    images = canonical_images("gqa_val_balanced")
    image = Image.open(PROJECT_ROOT / images[group[0]["sample_id"]])
    fig, axes = plt.subplots(1, len(group), figsize=(TEXT_IN, 2.35))
    for ax, r in zip(axes, group):
        overlay(ax, image, r["grid_hw"], set(r["keep_indices"]), set(r["pruner_metadata"]["focus_indices"]))
        question = r["question"].split("\n")[0]
        ax.set_title(f"{question}\n→ \"{r['prediction']}\"", fontsize=7.5, loc="left")
    fig.suptitle("DeSTiL-fc L5: same image, different questions → different focus regions (red outline); "
                 "other bright tokens are context", fontsize=8, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "fig_focus_questions.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "docs" / "paper" / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    for make in (fig_budget_curves, fig_mme_subtasks, fig_timing, fig_region, fig_ablation,
                 fig_selection, fig_focus_questions):
        make(args.output_dir)
        print(f"wrote {make.__name__}")


if __name__ == "__main__":
    main()
