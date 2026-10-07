---
name: python-testing
description: Build focused pytest coverage for loaders, configuration, and evaluation logic.
source: C:\Users\Admin\Desktop\ASkills\skills\python-testing-patterns\SKILL.md
---

# Python testing

- Test dataset configuration selection without downloading data.
- Test GQA balanced-only validation and rejection of `*_all_*` configs.
- Test config parsing, sample normalization, output paths, and metric helpers
  with small fixtures.
- Mock Hugging Face/model loading at unit-test boundaries.
- Keep at least one cheap end-to-end smoke test for the baseline runner when
  the environment supports it; do not require model weights in ordinary unit
  tests.
- Run focused checks with `uv run pytest`; report missing optional GPU/model
  dependencies instead of weakening assertions silently.
