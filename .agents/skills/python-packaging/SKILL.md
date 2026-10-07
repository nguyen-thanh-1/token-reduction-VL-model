---
name: python-packaging
description: Keep the installable research package and uv metadata reproducible.
source: C:\Users\Admin\Desktop\ASkills\skills\python-packaging\SKILL.md
---

# Python packaging

- Treat `pyproject.toml` as the source of truth for package metadata and
  dependencies.
- Keep the `src/` package discoverable through the configured build backend.
- Use lower bounds only when they reflect a real compatibility requirement;
  avoid unnecessary dependency churn.
- Change `pyproject.toml` and `uv.lock` together.
- Prefer `uv run ...` for commands so the lockfile and environment are used.
- Do not package data downloads, checkpoints, credentials, `.venv`, or local
  outputs.
- Before a release or handoff, verify the package can be imported from the
  project environment and report any validation not run.
