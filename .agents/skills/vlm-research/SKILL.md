---
name: vlm-research
description: Guide VLM/VQA and visual-token research decisions for the pruning study.
source: C:\Users\Admin\Desktop\ASkills\skills\computer-vision-expert\SKILL.md
---

# VLM research

Apply the VLM/VQA-relevant part of the computer-vision reference skill. The
current research question is adaptive visual-token reduction in a
vision-language model, not object detection or segmentation.

## Research rules

- Measure the full-token Qwen3-VL baseline before introducing pruning.
- Keep answer quality and efficiency as separate axes: accuracy, retained
  visual-token ratio, latency, throughput, VRAM, and FLOPs proxies where
  measurable.
- Preserve visual information needed for text-rich, chart, document, and
  spatial questions; do not assume attention magnitude alone is sufficient.
- Compare future methods against explicit controls such as random-K,
  uniform-K, and question-aware fixed-K.
- Make token budget, layer location, scoring signal, and spatial safeguards
  explicit in each experiment.
- Avoid claiming a method is state of the art without a reproducible baseline,
  dataset split, metric definition, and model revision.
