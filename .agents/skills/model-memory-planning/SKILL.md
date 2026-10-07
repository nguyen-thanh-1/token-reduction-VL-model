---
name: model-memory-planning
description: Estimate memory and KV-cache needs before local Hugging Face model runs.
source: C:\Users\Admin\Desktop\ASkills\skills\hf-mem\SKILL.md
---

# Model memory planning

Before downloading or loading a large checkpoint, estimate its memory needs
without pulling weights into the repository. The reference workflow uses
`uvx hf-mem` with a Hub model id and optionally `--experimental` for KV-cache
estimates.

For this project, record at least:

- model id and revision;
- dtype and quantization, if any;
- maximum sequence length and batch size;
- image resolution or processor settings;
- available GPU VRAM and system RAM;
- whether the run is full-token or pruned.

Treat memory estimates as planning data, not proof that a run will fit. Confirm
with a small smoke test and record the observed peak memory.
