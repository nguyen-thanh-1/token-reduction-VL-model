---
name: uv-python
description: Manage this project with uv and Python 3.12.
---

# uv and Python 3.12

- Use `.python-version` with Python `3.12`.
- Use `uv sync` to create or update the environment.
- Use `uv run ...` for project commands; do not mix system Python or pip
  installs into the project environment.
- Declare runtime dependencies in `pyproject.toml`.
- Regenerate `uv.lock` after dependency changes and review the diff.
- Keep `.venv/` and local uv caches out of version control.
- Do not change Python version constraints without a concrete compatibility
  reason.
