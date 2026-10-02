# Agent Working Guide

This repository contains Python 3.12 tooling for downloading and evaluating
vision-language model datasets.

## Required skills

Read the relevant skill before making changes:

- `skills/uv-python/SKILL.md` for Python, `uv`, dependencies, and lockfiles.
- `skills/huggingface-datasets/SKILL.md` for Hugging Face dataset code and
  storage rules.
- `skills/git-workflow/SKILL.md` for branches, commits, and remote changes.

## Working rules

1. Inspect the repository, current branch, and working tree before editing.
2. Keep changes focused and preserve unrelated user changes.
3. Do not download large datasets unless the user explicitly asks for it.
4. Keep downloaded datasets, `.venv`, uv caches, credentials, and local machine
   files out of Git.
5. Update `pyproject.toml` and `uv.lock` together when dependencies change.
6. Verify changes in proportion to their risk and report what was not run.
7. Never expose Hugging Face tokens or other credentials in source files,
   logs, commits, or documentation.

## Project commands

```powershell
uv sync
uv run python scripts/download_datasets.py --dataset gqa
uv run python scripts/download_datasets.py --dataset mmb
uv run python scripts/download_datasets.py --dataset mme
```

The downloader is separate from normal development commands so dependency
installation does not download the datasets.
