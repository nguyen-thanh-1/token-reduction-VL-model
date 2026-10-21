# DSFold proposal (not implemented)

This directory shows where the proposed DeepStack-aware Select-and-Fold
(DSFold) method would intervene in the full-token
`Qwen/Qwen3-VL-2B-Instruct` baseline.

- Editable source: `Qwen3-VL-2B-Instruct-DSFold-proposal.drawio`
- Derived from: `../model-goc-baseline/Qwen3-VL-2B-Instruct-architecture.drawio`
- Method write-up: `docs/research/dsfold-method-proposal.md`
- Image exports: `exports/`

The diagram is a copy of the single-page baseline diagram with these changes:

- Sections 3–5 and panel B are shifted right to make room for a new section,
  **2½. PROPOSED — DSFold**, between the vision encoder and multimodal fusion.
- Main visual embeddings and the three DeepStack features now flow through
  DSFold. DSFold outputs `K` kept tokens, three folded DeepStack tensors, and
  the kept indices with their original M-RoPE grid coordinates.
- Red outlines with `M1`–`M8` badges mark baseline cells whose behaviour
  changes; the top-right legend explains each one.
- Red `P0`–`P3` badges mark the candidate token-reduction points; the top-left
  legend explains them. Only P2 is the proposed method. P0 (resolution) and
  P3 (FastV-style) are comparison baselines.
- Panel **D** holds the algorithm, and box **⑦** in panel C holds the pruned
  token ledger.

Keep the baseline diagram unchanged; update this copy when the proposal changes.
