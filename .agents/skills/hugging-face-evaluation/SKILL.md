---
name: hugging-face-evaluation
description: Plan local model evaluation with explicit backends, smoke tests, and reproducible benchmark settings.
source: C:\Users\Admin\Desktop\ASkills\skills\hugging-face-community-evals\SKILL.md
---

# Hugging Face evaluation

This project has its own GQA/MMB/MME evaluation runner. Use this skill for
evaluation discipline, not as a replacement for the benchmark-specific
adapters.

## Workflow

1. Verify the environment and hardware before a local model run.
2. Start with a small limit, for example `--limit 10`.
3. Use Transformers as the compatibility baseline for Qwen3-VL; consider
   another backend only after measuring support and reproducibility.
4. Scale to the full benchmark only after the smoke test succeeds.
5. Save predictions and metrics separately, together with model revision,
   dataset configuration, prompt/generation settings, hardware, and git SHA.

## Failure handling

- On OOM, reduce sample limits/batch size or use a smaller smoke test first.
- If a backend does not support the model, fall back to Transformers rather
  than silently changing the experiment.
- Keep benchmark scoring separate from generation so the same predictions can
  be re-evaluated.
