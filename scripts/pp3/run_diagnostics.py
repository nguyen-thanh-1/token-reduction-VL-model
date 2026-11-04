"""Stage-1 PP3 diagnostics (ADR 0002): several selections on one split with one model load.

Each image is encoded once and every configuration below is decoded from the
same ViT output. Rows go to outputs/pp3_scorer/runs/<prefix>_<name>_<dataset>/<source>.jsonl
(dataset gqa, mmb or mme from --source; full tokens to outputs/none/runs/...),
in the run_pruned.py format, so
summarize_runs.py and the McNemar tools read them unchanged. Latency excludes
the ViT; use run_pruned.py for timing.

Configurations (<tag> names the score head, default 3a):
    full                    no pruning (reference on the same ids)
    <tag>_L3, <tag>_L4      PP3 as deployed (score head + DPP + fold)
Only with --only (budget curve, baselines, fold ablation):
    <tag>_L1, <tag>_L2, <tag>_L5        PP3 at 1/18, 1/9, 1/2 of the tokens
    <tag>_nofold_L1 ... _L5             PP3 selection, dropped tokens removed (gamma = 0)
    random_L1, random_L2, random_L5     random selection with fold
    divprune_L1 ... _L5                 DivPrune, no fold (ADR 0005)
    fastv_L1 ... _L5                    FastV-style, LLM layer-2 attention, no fold (ADR 0005)
    cdpruner_L1 ... _L5                 CDPruner, instruction-conditioned DPP, no fold (ADR 0005)
Default runs:
    <tag>_lowdrop10/25      drop the 10% / 25% lowest-q̂ tokens, no fold (denoising test)
    random_L3, random_L4    uniform random selection at the same K, with fold
    randdrop10/25           drop 10% / 25% random tokens, no fold   (noise floor)
    blind1                  one random token (near-blind language prior)

Runs that already have every row are skipped, so a second score head only
costs its own configurations.

Examples:
    uv run python scripts/pp3/run_diagnostics.py --limit 20
    uv run python scripts/pp3/run_diagnostics.py
    uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_signed.yaml         --tag 3a_signed --only 3a_signed_L3 3a_signed_L4 3a_signed_lowdrop10 3a_signed_lowdrop25
    uv run python scripts/pp3/run_diagnostics.py --method-config configs/methods/pp3_scorer_fc.yaml         --tag fc_b2 --fc beta=2 --only fc_b2_L3 fc_b2_L4
    uv run python scripts/pp3/run_diagnostics.py --only divprune_L2 divprune_L3 fastv_L2 fastv_L3
    uv run python scripts/pp3/run_diagnostics.py --source mmb_validation --split all --prefix final         --method-config configs/methods/pp3_scorer_attn50k.yaml --tag 3d_attn50k --only full 3d_attn50k_L3
"""

from __future__ import annotations

import argparse
import copy
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
from token_reduction_vl.pruning.pp3_scorer.dsfold_base import DSFoldBase  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.baselines import CDPrunerPruner, DivPrunePruner, FastVPruner  # noqa: E402
from token_reduction_vl.pruning.pp3_scorer.diagnostics import RandomKeepPruner, TopScorePruner  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "common"))
from run_pruned import finished_rows, load_pruner, prediction_record  # noqa: E402

BASELINE_CONFIG = PROJECT_ROOT / "configs" / "baseline_qwen3_vl.yaml"
DATASETS = {"gqa": "gqa", "mmb": "mmb", "mme": "mme"}
EXTRA_LEVELS = ("L1", "L2", "L5")


def dataset_suffix(source: str) -> str:
    """``gqa_val_balanced`` -> ``gqa``, ``mmb_validation`` -> ``mmb``, ``mme_test`` -> ``mme``."""

    return DATASETS.get(source.split("_", 1)[0], source)


def parse_overrides(text: str | None) -> dict[str, str]:
    """``"beta=2,share_max=0.9"`` -> ``{"beta": "2", "share_max": "0.9"}``."""

    if not text:
        return {}
    return dict(item.split("=", 1) for item in text.split(",") if item.strip())


def build_runs(method_config: str, tag: str, device: str, fc_overrides: dict[str, str] | None = None):
    """``(name, method_dir, variant, level, ratio, pruner, extra)`` for every configuration;
    ``extra`` runs only when named in --only."""

    _, scored, _ = load_pruner(method_config, device)
    if fc_overrides:
        if not hasattr(scored, "with_focus_params"):
            raise SystemExit("--fc needs a focus-context method config (e.g. configs/methods/pp3_scorer_fc.yaml).")
        scored = scored.with_focus_params(**fc_overrides)
    params = scored.base.params
    no_fold = params.with_overrides(gamma=0.0)
    top = TopScorePruner(scored.head, params=no_fold, compressor=scored.compressor)
    nofold_scored = copy.copy(scored)  # same selector, dropped tokens removed instead of folded
    nofold_scored.base = DSFoldBase(no_fold)
    divprune, fastv, cdpruner = DivPrunePruner(), FastVPruner(), CDPrunerPruner()
    runs = [
        ("full", "none", "full", "L7", None, NoOpPruner(), False),
        (f"{tag}_L3", "pp3_scorer", tag, "L3", BUDGET_LEVELS["L3"], scored, False),
        (f"{tag}_L4", "pp3_scorer", tag, "L4", BUDGET_LEVELS["L4"], scored, False),
        (f"{tag}_lowdrop10", "pp3_scorer", f"{tag}_lowdrop_nofold", "r0.90", 0.90, top, False),
        (f"{tag}_lowdrop25", "pp3_scorer", f"{tag}_lowdrop_nofold", "r0.75", 0.75, top, False),
        ("random_L3", "pp3_scorer", "random", "L3", BUDGET_LEVELS["L3"], RandomKeepPruner(params, seed=0), False),
        ("random_L4", "pp3_scorer", "random", "L4", BUDGET_LEVELS["L4"], RandomKeepPruner(params, seed=1), False),
        ("randdrop10", "pp3_scorer", "randdrop_nofold", "r0.90", 0.90, RandomKeepPruner(no_fold, seed=2), False),
        ("randdrop25", "pp3_scorer", "randdrop_nofold", "r0.75", 0.75, RandomKeepPruner(no_fold, seed=3), False),
        ("blind1", "pp3_scorer", "blind1", "K1", 1e-6, RandomKeepPruner(params, seed=4), False),
    ]
    for level in EXTRA_LEVELS:
        runs.append((f"{tag}_{level}", "pp3_scorer", tag, level, BUDGET_LEVELS[level], scored, True))
        runs.append((f"random_{level}", "pp3_scorer", "random", level, BUDGET_LEVELS[level],
                     RandomKeepPruner(params, seed=10 + int(level[1])), True))
    for level in ("L1", "L2", "L3", "L4", "L5"):
        ratio = BUDGET_LEVELS[level]
        runs.append((f"{tag}_nofold_{level}", "pp3_scorer", f"{tag}_nofold", level, ratio, nofold_scored, True))
        runs.append((f"divprune_{level}", "pp3_scorer", "divprune", level, ratio, divprune, True))
        runs.append((f"fastv_{level}", "pp3_scorer", "fastv", level, ratio, fastv, True))
        runs.append((f"cdpruner_{level}", "pp3_scorer", "cdpruner", level, ratio, cdpruner, True))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="gqa_val_balanced")
    parser.add_argument("--split", default=str(PROJECT_ROOT / "configs" / "splits" / "gqa_val_tune_2k.txt"),
                        help="Id list (configs/splits/*.txt) or 'all' for every sample of --source.")
    parser.add_argument("--prefix", default="tune2k")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method-config", default="configs/methods/pp3_scorer.yaml")
    parser.add_argument("--tag", default="3a", help="Name of the score head in run ids.")
    parser.add_argument("--only", nargs="+", default=None, help="Run only these configuration names.")
    parser.add_argument("--fc", default=None, help="Focus-context overrides, e.g. beta=2,share_max=0.9.")
    args = parser.parse_args()

    baseline = load_yaml(BASELINE_CONFIG)
    model_config = baseline["model"]
    runner = PrunedQwen3VL(
        model_config["id"], dtype=model_config.get("dtype"), device_map=model_config.get("device_map", "auto"),
        attn_implementation=model_config.get("attn_implementation"),
        max_new_tokens=model_config.get("max_new_tokens", 32),
    )
    runs = build_runs(args.method_config, args.tag, str(runner.device), parse_overrides(args.fc))
    if args.only:
        unknown = set(args.only) - {run[0] for run in runs}
        if unknown:
            raise SystemExit(f"Unknown configurations {sorted(unknown)}; choose from {[run[0] for run in runs]}")
        runs = [run for run in runs if run[0] in args.only]
    else:
        runs = [run for run in runs if not run[6]]
    suffix = dataset_suffix(args.source)
    method_config = load_yaml(PROJECT_ROOT / args.method_config)

    paths, done = {}, {}
    for name, method_dir, *_ in runs:
        run_dir = PROJECT_ROOT / "outputs" / method_dir / "runs" / f"{args.prefix}_{name}_{suffix}"
        run_dir.mkdir(parents=True, exist_ok=True)
        paths[name] = run_dir / f"{args.source}.jsonl"
        previous = finished_rows(paths[name])
        paths[name].write_text("".join(line + "\n" for line in previous), encoding="utf-8")
        done[name] = {json.loads(line)["sample_id"] for line in previous}

    source = baseline["data"]["sources"][args.source]
    if args.split == "all":
        with (PROJECT_ROOT / source["path"]).open(encoding="utf-8") as handle:
            ids = [json.loads(line)["sample_id"] for line in handle if line.strip()]
    else:
        ids = read_split(Path(args.split))
    ids = ids[: args.limit] if args.limit else ids
    pending = {i for i in ids if any(i not in done[name] for name, *_ in runs)}
    samples = iter_split_samples(PROJECT_ROOT / source["path"], pending, project_root=PROJECT_ROOT)
    started_at = utc_now()
    handles = {name: paths[name].open("a", encoding="utf-8") for name, *_ in runs}
    try:
        for sample in tqdm(samples, total=len(pending), desc="diagnostics"):
            scoring_text = question_text_for_scoring(sample.question, sample.metadata.get("choices") or [])
            encoded = runner.encode(sample.image, sample.question, scoring_text)
            prepared_pruners: set[int] = set()
            for name, _, variant, level, ratio, pruner, _ in runs:
                if sample.sample_id in done[name]:
                    continue
                if hasattr(pruner, "prepare_sample") and id(pruner) not in prepared_pruners:
                    pruner.prepare_sample(runner, encoded)
                    prepared_pruners.add(id(pruner))
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

    for name, method_dir, variant, level, ratio, _, _ in runs:
        rows = len(finished_rows(paths[name]))
        write_run_meta(
            paths[name].parent / "run_meta.json",
            project_root=PROJECT_ROOT,
            run_id=paths[name].parent.name,
            method="none" if name == "full" else "pp3_scorer",
            variant=variant,
            level=level,
            source=args.source,
            split=args.split,
            limit=args.limit,
            config="none" if name == "full" else (
                f"{args.method_config} ({name}; scripts/pp3/run_diagnostics.py"
                + (f"; --fc {args.fc})" if args.fc else ")")
            ),
            dsfold_params=method_config.get("dsfold_params"),
            checkpoint=None if name in ("full",) or name.startswith(("random", "randdrop", "blind", "divprune", "fastv", "cdpruner"))
            else method_config.get("checkpoint"),
            model_id=model_config["id"],
            started_at=started_at,
            finished_at=utc_now(),
            rows=rows,
            errors=0,
        )
        print(f"{name:<12} {rows:>5} rows -> {paths[name]}")


if __name__ == "__main__":
    main()
