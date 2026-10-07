---
name: hugging-face-cli
description: Use Hugging Face Hub CLI and environment variables safely for model and dataset operations.
source: C:\Users\Admin\Desktop\ASkills\skills\hugging-face-cli\SKILL.md
---

# Hugging Face Hub operations

Use the current `hf` command when Hub inspection, authentication, cache
management, or explicit file/model operations are needed. The deprecated
`huggingface-cli` name should not be introduced.

## Project guidance

- Prefer the repository's downloader for the three benchmark datasets so the
  GQA balanced-only restriction remains enforced.
- Use `HF_TOKEN` or an interactive `hf auth login`; never write a token into
  source, config, logs, or README examples.
- Inspect a repo or use a dry run before a large download when possible.
- Keep Hub cache and downloaded content outside Git-tracked source; the project
  stores completed datasets under `data/raw/`.
- Pin or record a Hub revision for experiments that need reproducibility.
- Treat upload, delete, branch, and repo-management commands as explicit
  external actions requiring user intent.

Useful read-only commands include `hf env`, `hf auth whoami`, `hf datasets
info`, `hf models info`, and `hf cache list`.
