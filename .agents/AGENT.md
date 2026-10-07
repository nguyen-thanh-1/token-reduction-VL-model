# Agent Working Guide

This repository contains Python 3.12 tooling for downloading and evaluating
vision-language model datasets, with a full-token Qwen3-VL baseline and future
visual-token pruning experiments.

## Required skills

Read the relevant skill before making changes:

- `skills/uv-python/SKILL.md` for Python, `uv`, dependencies, and lockfiles.
- `skills/huggingface-datasets/SKILL.md` for Hugging Face dataset code and
  storage rules.
- `skills/git-workflow/SKILL.md` for branches, commits, and remote changes.
- `skills/python-project-scaffold/SKILL.md` for repository architecture,
  `src/` layout, and development tooling.
- `skills/python-packaging/SKILL.md` for `pyproject.toml`, package metadata,
  and reproducible builds.
- `skills/hugging-face-cli/SKILL.md` for Hugging Face authentication, cache,
  model, and dataset operations.
- `skills/hugging-face-evaluation/SKILL.md` for local evaluation strategy,
  smoke tests, and backend selection.
- `skills/vlm-research/SKILL.md` for VLM/VQA and visual-token research
  decisions.
- `skills/model-memory-planning/SKILL.md` before planning a local model run or
  estimating GPU/CPU memory.
- `skills/research-decision-records/SKILL.md` when selecting or changing a
  pruning method, evaluation protocol, or major project architecture.
- `skills/python-testing/SKILL.md` when adding tests or test infrastructure.

The skills listed after the project-specific skills were adapted from the
read-only reference collection at `C:\Users\Admin\Desktop\ASkills`. The source
collection must not be edited from this repository. See
`ASKILLS_SELECTION.md` for the selection rationale and source mapping.

## Working rules

1. Inspect the repository, current branch, and working tree before editing.
2. Keep changes focused and preserve unrelated user changes.
3. Do not download large datasets or model weights unless the user explicitly
   asks for it.
4. Keep downloaded datasets, `.venv`, uv caches, credentials, and local machine
   files out of Git.
5. Update `pyproject.toml` and `uv.lock` together when dependencies change.
6. Verify changes in proportion to their risk and report what was not run.
7. Never expose Hugging Face tokens or other credentials in source files,
   logs, commits, or documentation.
8. Keep the baseline method-agnostic until an experiment or ADR justifies a
   pruning choice; do not present a candidate method as the final method.
9. Start evaluation with a small smoke test before a full benchmark run.
10. Record benchmark configuration, model revision, dataset configuration,
    generation settings, hardware, and git revision with every experiment.

## Project commands

```powershell
uv sync
uv run python scripts/download_datasets.py --dataset gqa
uv run python scripts/download_datasets.py --dataset mmb
uv run python scripts/download_datasets.py --dataset mme
uv run python scripts/run_baseline.py --dataset mme_test --limit 10
```

The downloader is separate from normal development commands so dependency
installation does not download datasets or model weights. The current baseline
is intentionally full-token and method-agnostic; pruning implementations can
be added under `src/token_reduction_vl/pruning/` after the baseline is measured.
