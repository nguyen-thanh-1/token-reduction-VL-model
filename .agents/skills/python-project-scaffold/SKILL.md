---
name: python-project-scaffold
description: Maintain a research-ready Python project structure with uv, src layout, scripts, configuration, and tests.
source: C:\Users\Admin\Desktop\ASkills\skills\python-development-python-scaffold\SKILL.md
---

# Python project scaffold

Use this skill when adding a package, script, configuration, or test to the
research repository.

## Project rules

- Keep reusable code under `src/token_reduction_vl/`.
- Keep runnable entry points under `scripts/` and keep them thin.
- Keep experiment configuration in `configs/`, not hard-coded in model code.
- Put tests under `tests/` and make them runnable with `uv run pytest`.
- Use type hints and small, composable functions for data and evaluation code.
- Prefer the existing generic research layout over introducing a web framework.

## Checklist

1. Inspect the current tree before creating a new top-level directory.
2. Update `pyproject.toml` for runtime dependencies and regenerate `uv.lock`.
3. Add a smoke-testable entry point and document its command in `README.md`.
4. Keep datasets, model weights, caches, and outputs outside the tracked source.
5. Validate imports/AST and run focused tests when dependencies are available.
