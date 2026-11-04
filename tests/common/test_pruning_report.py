import json

from PIL import Image

from token_reduction_vl.reporting.pruning_report import (
    collect_summaries,
    plot_summaries,
    plot_token_usage,
    render_kept_tokens,
    write_tables,
)


def gqa_row(sample_id: str, prediction: str, answer: str, **extra) -> dict:
    return {
        "sample_id": sample_id, "dataset": "gqa", "split": "testdev", "task_type": "open_vqa",
        "question": "q?\n\nAnswer with a short phrase.", "references": [answer], "prediction": prediction,
        "input_tokens": 300, "metadata": {"dataset": "gqa"}, **extra,
    }


def write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_runs_are_compared_with_the_baseline_on_the_same_ids(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    write_jsonl(baseline / "gqa_testdev_balanced.jsonl", [
        gqa_row("a", "yes", "yes"), gqa_row("b", "red", "red"), gqa_row("c", "no", "yes"), gqa_row("d", "cat", "cat"),
    ])
    run = tmp_path / "runs" / "r1"
    timing = {"vit": 0.05, "selector": 0.01, "prefill": 0.02, "decode": 0.03}
    write_jsonl(run / "gqa_testdev_balanced.jsonl", [
        gqa_row("a", "yes", "yes", method="pp3_scorer", variant="v", level="L3", N=100, K=22, input_tokens=48, timing=timing),
        gqa_row("b", "blue", "red", method="pp3_scorer", variant="v", level="L3", N=100, K=23, input_tokens=49, timing=timing),
    ])
    (summary,) = collect_summaries([run], baseline)
    assert summary["accuracy"] == 0.5
    assert summary["baseline_accuracy"] == 1.0  # only ids a and b, not the whole baseline file
    assert summary["agree_with_baseline"] == 0.5
    assert summary["keep_ratio_realized"] == 0.225
    assert summary["baseline_input_tokens"] == 300

    write_tables([summary], tmp_path / "report")
    assert "pp3_scorer" in (tmp_path / "report" / "summary.md").read_text()
    assert all(path.exists() for path in plot_summaries([summary], tmp_path / "report"))
    assert plot_token_usage([run / "gqa_testdev_balanced.jsonl"], tmp_path / "report")


def test_render_kept_tokens_darkens_dropped_cells() -> None:
    image = Image.new("RGB", (100, 60), color=(200, 200, 200))
    rendered = render_kept_tokens(image, (2, 3), [0, 4])
    assert rendered.size == (96, 64)
    kept_pixel = rendered.getpixel((16, 16))  # inside token 0
    dropped_pixel = rendered.getpixel((48, 16))  # inside token 1
    assert sum(kept_pixel) > 2 * sum(dropped_pixel)
