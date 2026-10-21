# CLAUDE.md

Thesis research codebase: adaptive visual-token pruning for vision-language
models. A full-token Qwen3-VL-2B-Instruct baseline is measured on GQA, MMBench,
and MME; no pruning method has been chosen yet. The user usually writes in
Vietnamese; reply in Vietnamese, keep code and repo docs in English.

`AGENT.md` and `.agents/AGENT.md` are the shared agent guides (also used by
other agents); this file adds Claude-specific routing. If they disagree, ask.

## Commands

Always go through `uv` with Python 3.12 (Windows, PowerShell).

```powershell
uv sync                                   # env from uv.lock (CUDA 13.0 torch wheels)
uv run pytest -q                          # 18 CPU-only tests, ~20 s, no downloads
uv run python scripts/preprocess_datasets.py --dataset all --limit 10   # smoke
uv run python scripts/run_baseline.py --dataset mme_test --limit 10     # GPU smoke
uv run python scripts/evaluate_predictions.py outputs/baseline/mme_test.jsonl --output data/metrics/mme_test.json
uv run python scripts/visualize_benchmarks.py [--overwrite]
uv run python scripts/generate_failure_gallery.py
```

Never start dataset downloads (`scripts/download_datasets.py`), model weight
downloads, full preprocessing, or full GPU benchmark runs unless the user asks
in this conversation. Unit tests must stay free of model weights and network.

## Layout and data flow

```
download (data/raw/{GQA,MMB,MME}) -> preprocess (data/processed/v1/*.jsonl)
  -> run_baseline (outputs/baseline/<source>.jsonl) -> evaluate_predictions
  -> visualize_benchmarks / generate_failure_gallery (outputs/reports/...)
```

- `src/token_reduction_vl/` — all reusable code; `scripts/` stay thin CLIs.
  - `data/` downloader, source adapters (`adapters/gqa|mmb|mme.py`), canonical
    schema (`schema.py`, `canonical.py`), runtime Qwen messages (`messages.py`).
  - `models/qwen3_vl.py` — `Qwen3VLBaseline` (Transformers `generate`).
  - `evaluation/` — `run_baseline` JSONL writer, benchmark-aware `metrics.py`.
  - `pruning/base.py` — `VisualTokenPruner` protocol + `NoOpPruner`; new
    methods go here.
  - `reporting/` — figures, tables, `report.md`, HTML failure gallery.
- `configs/` — single source of truth: `baseline_qwen3_vl.yaml` (model, run
  sources), `data_v1.yaml` (split roles), `report_baseline.yaml` (report).
- `docs/adr/` — research decision records. `architecture/` — Draw.io diagrams.

## Hard rules

- GQA: balanced configurations only; never add or fall back to `*_all_*`.
- Split roles (`configs/data_v1.yaml`): train GQA train, tune GQA val, evaluate
  GQA testdev + MMBench validation + MME test. GQA test and MMBench test are
  inference-only — never report accuracy on them.
- Never average GQA, MMBench, and MME accuracies; MME score is not a percentage.
- Persisted data stays model-agnostic (no Qwen special tokens in JSONL).
- Keep the baseline method-agnostic; a pruning choice needs an ADR first.
- A new method gets its own `report.id`/`output_dir`; do not overwrite the
  full-token baseline report.
- Every experiment records config, model/dataset revision, generation
  settings, hardware, and git SHA. Report quality and efficiency separately.
- `outputs/` predictions and reports ARE tracked in Git; `data/`, checkpoints,
  `.venv`, `.uv-cache`, `.codegraph/` are not.
- Change `pyproject.toml` and `uv.lock` together. Never expose `HF_TOKEN`.
- Git: feature branches off `dev`; commit/push/PR only when asked.

## Skills — read before working in that area

Project skills live in `.agents/skills/` (not auto-loaded by Claude Code).
Read the matching `SKILL.md` before changing code in that area:

| Task | Skill |
| --- | --- |
| Any Python command, dependency, `uv.lock` | `uv-python` |
| `pyproject.toml`, packaging, build | `python-packaging` |
| New module, script, config, directory | `python-project-scaffold` |
| Downloader, dataset sources, `data/raw` | `huggingface-datasets` |
| `hf` CLI, auth, cache, Hub revisions | `hugging-face-cli` |
| Baseline inference, scoring, scaling a run | `hugging-face-evaluation` |
| New checkpoint, dtype, resolution, hardware | `model-memory-planning` |
| Pruning design/implementation, result analysis | `vlm-research` |
| Choosing/changing method, metric, protocol | `research-decision-records` |
| Tests and fixtures | `python-testing` |
| Branches, commits, pushes, PRs | `git-workflow` |

`.agents/ASKILLS_SELECTION.md` records where adapted skills came from
(`C:\Users\Admin\Desktop\ASkills`, read-only — never edit it).

## CodeGraph

The repo is indexed (`.codegraph/`, local only). Use `codegraph_explore` with
`projectPath` set to this repo root before Grep/Read to locate code or check a
symbol's blast radius. Shell fallback: `codegraph explore "<symbols>"`. After
large edits or branch switches, run `codegraph sync`.

## Architecture diagram

The baseline diagram is the single-page
`architecture/model-goc-baseline/Qwen3-VL-2B-Instruct-architecture.drawio`.
When a diagram is added, renamed, or restructured, update `README.md`,
`architecture/README.md`, and the per-model README together.
