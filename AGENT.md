# Agent Working Guide

This repository develops Python 3.12 tooling for VLM dataset preparation,
Qwen3-VL baseline evaluation, and future adaptive visual-token pruning
experiments.

The detailed project instructions and skill catalog are maintained in:

- `.agents/AGENT.md`
- `.agents/ASKILLS_SELECTION.md`
- `.agents/skills/`

## Essential rules

- Use `uv` and Python 3.12 for project commands.
- Read the relevant skill in `.agents/skills/` before changing code.
- Keep reusable code under `src/token_reduction_vl/` and runnable entry points
  under `scripts/`.
- Keep datasets, model weights, caches, credentials, `.venv`, and generated
  outputs out of Git.
- Preserve the GQA balanced-only dataset restriction.
- Measure the full-token Qwen3-VL baseline before implementing a pruning method.
- Start model evaluation with a small smoke test and record configuration,
  model revision, hardware, and git revision.
- Do not expose Hugging Face tokens in source files, logs, commits, or docs.

## Common commands

```powershell
uv sync
uv run python scripts/download_datasets.py --dataset gqa
uv run python scripts/download_datasets.py --dataset mmb
uv run python scripts/download_datasets.py --dataset mme
uv run python scripts/run_baseline.py --dataset mme_test --limit 10
```

Do not run large dataset or model downloads unless explicitly requested.
