"""Stage-1 PP3 diagnostics (ADR 0002): several selections on one split with one model load.

Each image is encoded once and every configuration below is decoded from the
same ViT output. Rows go to outputs/pp3_scorer/runs/<prefix>_<name>_gqa/<source>.jsonl
(full tokens to outputs/none/runs/...), in the run_pruned.py format, so
summarize_runs.py and the McNemar tools read them unchanged. Latency excludes
the ViT; use run_pruned.py for timing.

Configurations:
    full                   no pruning (reference on the same ids)
    3a_L3, 3a_L4           PP3 as deployed (score head + DPP + fold)
    random_L3, random_L4   uniform random selection at the same K, with fold
    randdrop10/25          drop 10% / 25% random tokens, no fold   (noise floor)
    lowdrop10/25           drop the 10% / 25% lowest-q̂ tokens, no fold (denoising test)
    blind1                 one random token (near-blind language prior)

Example:
    uv run python scripts/pp3/run_diagnostics.py --limit 20
    uv run python scripts/pp3/run_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tqdm import tqdm  # noqa: E402

from token_reduction_vl.config import load_yaml  # noqa: E402
from token_reduction_vl.data.splits import iter_split_samples, read_split  # noqa: E402
from token_reduction_vl.evaluation.run_meta import utc_now, write_run_meta  # noqa: E402
from token_reduction_vl.models.qwen3_vl_pruned import PrunedQwen3VL, question_text_for_scoring  # noqa: E402
from token_reduction_vl.pruning.base import BUDGET_LEVELS, NoOpPruner  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.diagnostics import RandomKeepPruner, TopScorePruner  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "common"))
from run_pruned import finished_rows, load_pruner, prediction_record  # noqa: E402

BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline_qwen3_vl.yaml"
METHOD_CONFIG = "configs/methods/pp3_scorer.yaml"


def build_runs(device: str):
    """``(name, method_dir, variant, level, ratio, pruner)`` for every configuration."""

    _, scored, _ = load_pruner(METHOD_CONFIG, device)
    params = scored.base.params
    no_fold = params.with_overrides(gamma=0.0)
    top = TopScorePruner(scored.head, params=no_fold, compressor=scored.compressor)
    runs = [
        ("full", "none", "full", "L7", None, NoOpPruner()),
        ("3a_L3", "pp3_scorer", "3a_s0", "L3", BUDGET_LEVELS["L3"], scored),
        ("3a_L4", "pp3_scorer", "3a_s0", "L4", BUDGET_LEVELS["L4"], scored),
        ("random_L3", "pp3_scorer", "random", "L3", BUDGET_LEVELS["L3"], RandomKeepPruner(params, seed=0)),
        ("random_L4", "pp3_scorer", "random", "L4", BUDGET_LEVELS["L4"], RandomKeepPruner(params, seed=1)),
        ("randdrop10", "pp3_scorer", "randdrop_nofold", "r0.90", 0.90, RandomKeepPruner(no_fold, seed=2)),
        ("randdrop25", "pp3_scorer", "randdrop_nofold", "r0.75", 0.75, RandomKeepPruner(no_fold, seed=3)),
        ("lowdrop10", "pp3_scorer", "lowdrop_nofold", "r0.90", 0.90, top),
        ("lowdrop25", "pp3_scorer", "lowdrop_nofold", "r0.75", 0.75, top),
        ("blind1", "pp3_scorer", "blind1", "K1", 1e-6, RandomKeepPruner(params, seed=4)),
    ]
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="gqa_val_balanced")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "configs" / "splits" / "gqa_val_tune_2k.txt")
    parser.add_argument("--prefix", default="tune2k")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    baseline = load_yaml(BASELINE_CONFIG)
    model_config = baseline["model"]
    runner = PrunedQwen3VL(
        model_config["id"], dtype=model_config.get("dtype"), device_map=model_config.get("device_map", "auto"),
        attn_implementation=model_config.get("attn_implementation"),
        max_new_tokens=model_config.get("max_new_tokens", 32),
    )
    runs = build_runs(str(runner.device))
    method_config = load_yaml(PROJECT_ROOT / METHOD_CONFIG)

    paths, done = {}, {}
    for name, method_dir, *_ in runs:
        run_dir = PROJECT_ROOT / "outputs" / method_dir / "runs" / f"{args.prefix}_{name}_gqa"
        run_dir.mkdir(parents=True, exist_ok=True)
        paths[name] = run_dir / f"{args.source}.jsonl"
        previous = finished_rows(paths[name])
        paths[name].write_text("".join(line + "\n" for line in previous), encoding="utf-8")
        done[name] = {json.loads(line)["sample_id"] for line in previous}

    ids = read_split(args.split)[: args.limit] if args.limit else read_split(args.split)
    pending = {i for i in ids if any(i not in done[name] for name, *_ in runs)}
    source = baseline["data"]["sources"][args.source]
    samples = iter_split_samples(PROJECT_ROOT / source["path"], pending, project_root=PROJECT_ROOT)
    started_at = utc_now()
    handles = {name: paths[name].open("a", encoding="utf-8") for name, *_ in runs}
    try:
        for sample in tqdm(samples, total=len(pending), desc="diagnostics"):
            scoring_text = question_text_for_scoring(sample.question, sample.metadata.get("choices") or [])
            encoded = runner.encode(sample.image, sample.question, scoring_text)
            for name, _, variant, level, ratio, pruner in runs:
                if sample.sample_id in done[name]:
                    continue
                result = runner.predict(sample.image, sample.question, scoring_text, pruner, ratio, encoded=encoded)
                record = prediction_record(
                    sample, result, method="none" if name == "full" else "pp3_scorer", variant=variant,
                    level=level, ratio=ratio, save_keep=True,
                )
                handles[name].write(json.dumps(record, ensure_ascii=False) + "\n")
                handles[name].flush()
            del encoded
    finally:
        for handle in handles.values():
            handle.close()

    for name, method_dir, variant, level, ratio, _ in runs:
        rows = len(finished_rows(paths[name]))
        write_run_meta(
            paths[name].parent / "run_meta.json",
            project_root=PROJECT_ROOT,
            run_id=paths[name].parent.name,
            method="none" if name == "full" else "pp3_scorer",
            variant=variant,
            level=level,
            source=args.source,
            split=str(args.split),
            limit=args.limit,
            config="none" if name == "full" else f"{METHOD_CONFIG} ({name}; scripts/pp3/run_diagnostics.py)",
            dsfold_params=method_config.get("dsfold_params"),
            checkpoint=None if name in ("full",) or name.startswith(("random", "randdrop", "blind")) else method_config.get("checkpoint"),
            model_id=model_config["id"],
            started_at=started_at,
            finished_at=utc_now(),
            rows=rows,
            errors=0,
        )
        print(f"{name:<12} {rows:>5} rows -> {paths[name]}")


if __name__ == "__main__":
    main()
