"""PP3 step 2a: fit the fixed PCA compressor on the first training samples.

Collects a random subset of token rows per sample (v, standardized DeepStack,
question embeddings) and saves data/features/pp3_scorer/compressor.npz.

Example:
    uv run python scripts/pp3/fit_compressor.py --config configs/methods/pp3_scorer.yaml
"""

from __future__ import annotations

import argparse
import sys
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
from token_reduction_vl.pruning.pp3_scorer.dsfold_base import DSFoldBase, load_dsfold_params  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.features import FeatureCompressor, question_rows  # noqa: E402

ROWS_PER_SAMPLE = 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "methods" / "pp3_scorer.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    fit = config["compressor_fit"]
    model = config["model"]
    runner = PrunedQwen3VL(model["id"], dtype=model.get("dtype"), device_map=model.get("device_map", "auto"))
    base = DSFoldBase(load_dsfold_params(PROJECT_ROOT / config["dsfold_params"]))
    ids = read_split(PROJECT_ROOT / config["labels"]["split"])[: fit["num_samples"]]

    generator = torch.Generator().manual_seed(0)
    v_rows, level_rows, q_rows = [], [[], [], []], []
    samples = iter_split_samples(PROJECT_ROOT / config["labels"]["canonical"], ids, project_root=PROJECT_ROOT)
    for sample in tqdm(samples, total=len(ids), desc="fit-compressor"):
        text = question_text_for_scoring(sample.question, sample.metadata.get("choices") or [])
        encoded = runner.encode(sample.image, sample.question, text)
        x = runner.pruner_input(encoded, 1.0)
        scores = base.scores(x)
        pick = torch.randperm(x.num_tokens, generator=generator)[:ROWS_PER_SAMPLE].to(scores.v_unit.device)
        v_rows.append(scores.v_unit[pick].cpu())
        for level, rows in zip(scores.f_hat, level_rows, strict=True):
            rows.append(level[pick].cpu())
        q_rows.append(question_rows(x, base.params.stopwords).cpu())

    compressor = FeatureCompressor.fit(
        torch.cat(v_rows),
        [torch.cat(rows) for rows in level_rows],
        torch.cat(q_rows),
        v_dim=fit["v_dim"],
        level_dim=fit["level_dim"],
        q_dim=fit["q_dim"],
    )
    output = PROJECT_ROOT / config["compressor"]
    compressor.save(output)
    print(f"Saved compressor {compressor.dims} to {output}")


if __name__ == "__main__":
    main()
