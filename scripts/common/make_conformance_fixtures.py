"""Create the PP1 conformance fixtures and, optionally, the reference outputs.

Step 1 (GPU, once, by one member): encode the 100 samples of
configs/splits/pp1_conformance_100.txt and save fp16 tensors to
data/features/pp1_conformance/ (not tracked by Git; share the folder).

Step 2 (CPU): --write-reference computes keep_indices at L5/L3/L1 with one
reviewed DSFold copy and writes tests/fixtures/pp1_conformance/reference.json.
Afterwards every member runs:  uv run pytest tests/test_pp1_conformance.py

Examples:
    uv run python scripts/common/make_conformance_fixtures.py
    uv run python scripts/common/make_conformance_fixtures.py --write-reference --impl pp3_scorer
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tqdm import tqdm  # noqa: E402

from token_reduction_vl.config import load_yaml  # noqa: E402
from token_reduction_vl.data.splits import iter_split_samples, read_split  # noqa: E402
from token_reduction_vl.pruning.base import BUDGET_LEVELS  # noqa: E402
from token_reduction_vl.pruning.conformance import load_fixture, save_fixture  # noqa: E402

FIXTURES = PROJECT_ROOT / "data" / "features" / "pp1_conformance"
REFERENCE = PROJECT_ROOT / "tests" / "fixtures" / "pp1_conformance" / "reference.json"
CANONICAL = ("gqa/testdev.jsonl", "mmbench/validation.jsonl", "mme/test.jsonl")
REFERENCE_LEVELS = ("L5", "L3", "L1")


def encode_fixtures(split: Path) -> None:
    from token_reduction_vl.models.qwen3_vl_pruned import PrunedQwen3VL, question_text_for_scoring

    model = load_yaml(PROJECT_ROOT / "configs" / "baseline_qwen3_vl.yaml")["model"]
    runner = PrunedQwen3VL(model["id"], dtype=model.get("dtype"), device_map=model.get("device_map", "auto"))
    ids = read_split(split)
    for relative in CANONICAL:
        samples = iter_split_samples(PROJECT_ROOT / "data" / "processed" / "v1" / relative, ids, project_root=PROJECT_ROOT)
        for sample in tqdm(samples, desc=relative):
            text = question_text_for_scoring(sample.question, sample.metadata.get("choices") or [])
            encoded = runner.encode(sample.image, sample.question, text)
            path = FIXTURES / fixture_name(sample.sample_id)
            save_fixture(path, runner.pruner_input(encoded, None), sample_id=sample.sample_id)
    print(f"Saved fixtures to {FIXTURES}")


def fixture_name(sample_id: str) -> str:
    """Flat, filesystem-safe file name; MME ids contain '/' and ':'."""

    return re.sub(r"[^A-Za-z0-9._-]", "_", sample_id) + ".npz"


def write_reference(impl: str) -> None:
    base_cls = importlib.import_module(f"token_reduction_vl.pruning.{impl}.dsfold_base").DSFoldBase
    entries = []
    paths = sorted(FIXTURES.glob("*.npz"))
    nested = [path for path in FIXTURES.rglob("*.npz") if path.parent != FIXTURES]
    if nested:
        raise SystemExit(f"{len(nested)} fixtures are in sub-folders (old naming); re-run the encoding step first.")
    for path in tqdm(paths, desc="reference"):
        for level in REFERENCE_LEVELS:
            sample_id, x = load_fixture(path, BUDGET_LEVELS[level])
            out = base_cls().select(x)
            entries.append({
                "file": path.name,
                "sample_id": sample_id,
                "level": level,
                "N": x.num_tokens,
                "keep_indices": out.keep_indices.tolist(),
                "deepstack_sum": [float(level_out.double().sum()) for level_out in out.deepstack_kept],
            })
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(json.dumps({"implementation": impl, "entries": entries}, indent=1), encoding="utf-8")
    print(f"Wrote {len(entries)} reference entries from {impl} to {REFERENCE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "configs" / "splits" / "pp1_conformance_100.txt")
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--impl", default="pp3_scorer", help="Package whose dsfold_base produces the reference.")
    args = parser.parse_args()
    if args.write_reference:
        write_reference(args.impl)
    else:
        encode_fixtures(args.split)


if __name__ == "__main__":
    main()
