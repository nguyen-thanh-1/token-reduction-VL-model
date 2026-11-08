"""Qwen3-VL-4B port (ADR 0007): config consistency and run-directory separation."""

from __future__ import annotations

from pathlib import Path

import pytest

from token_reduction_vl.config import load_yaml
from token_reduction_vl.models.qwen3_vl_pruned import check_run_prefix

ROOT = Path(__file__).resolve().parents[2]
CONFIG_2B = load_yaml(ROOT / "configs" / "methods" / "pp3_scorer_attn50k.yaml")
CONFIG_4B = load_yaml(ROOT / "configs" / "methods" / "pp3_scorer_attn_4b.yaml")
BASELINE = load_yaml(ROOT / "configs" / "baseline_qwen3_vl.yaml")["model"]


def test_4b_pins_the_checkpoint_and_keeps_the_method() -> None:
    model = CONFIG_4B["model"]
    assert model["id"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert len(model["revision"]) == 40
    assert CONFIG_4B["dsfold_params"] == CONFIG_2B["dsfold_params"]
    assert CONFIG_4B["compressor_fit"] == CONFIG_2B["compressor_fit"]
    route_2b = {k: v for k, v in CONFIG_2B["route_b"].items() if k != "name"}
    route_4b = {k: v for k, v in CONFIG_4B["route_b"].items() if k != "name"}
    assert route_4b == route_2b
    assert CONFIG_4B["labels"]["split"] == CONFIG_2B["labels"]["split"]


def test_4b_teacher_layers_keep_the_relative_depth() -> None:
    layers_2b, layers_4b = CONFIG_2B["labels"]["layers"], CONFIG_4B["labels"]["layers"]
    assert layers_4b == list(range(layers_4b[0], layers_4b[-1] + 1))
    assert layers_4b[0] == round(layers_2b[0] * 36 / 28)
    assert layers_4b[-1] == int(layers_2b[-1] * 36 / 28)
    assert layers_4b[-1] < 36


def test_4b_never_shares_a_path_with_2b() -> None:
    for key in ("compressor", "checkpoint"):
        assert CONFIG_4B[key] != CONFIG_2B[key]
    assert CONFIG_4B["labels"]["output_dir"] != CONFIG_2B["labels"]["output_dir"]
    assert CONFIG_4B["labels"]["prefix"] != CONFIG_2B["labels"]["prefix"]
    assert CONFIG_4B["route_b"]["name"] != CONFIG_2B["route_b"]["name"]


def test_run_prefix_guard() -> None:
    check_run_prefix(CONFIG_2B["model"], BASELINE, "final")  # the baseline model: any prefix
    check_run_prefix(CONFIG_4B["model"], BASELINE, "final4b")
    with pytest.raises(ValueError, match="final4b"):
        check_run_prefix(CONFIG_4B["model"], BASELINE, "final")
    with pytest.raises(ValueError, match="run_tag"):
        check_run_prefix({"id": "Qwen/Qwen3-VL-4B-Instruct"}, BASELINE, "final4b")
