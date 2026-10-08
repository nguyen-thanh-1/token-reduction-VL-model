# Model gốc (baseline)

This directory documents the full-token `Qwen/Qwen3-VL-2B-Instruct` baseline
used by the project.

- Editable source: `qwen3-vl-2b-instruct-baseline.drawio`
- Image exports: `exports/`
- Hugging Face snapshot inspected:
  `89644892e4d85e24eaac8bacfd4f463576704203`
- Local Transformers implementation inspected: `5.19.0`
- Runtime configuration: `configs/baseline_qwen3_vl.yaml`

The Draw.io file contains three pages:

1. **Clean overview** — a low-crossing, left-to-right view of input processing,
   both encoder paths, multimodal fusion, decoding, and evaluation.
2. **Vision encoder detail** — patch embedding, position encoding, all 24
   Vision Transformer blocks, DeepStack taps, the internal VisionBlock, and
   the 2×2 PatchMerger.
3. **Fusion and decoder detail** — masked-scatter fusion, M-RoPE, all 28
   decoder layers, three DeepStack additions, grouped-query attention, SwiGLU,
   tied LM head, and generation.

The baseline retains every visual token produced by the built-in spatial patch
merger. The built-in merger is documented separately from future experimental
token pruning to avoid conflating the two mechanisms.
