# Model gốc (baseline)

This directory documents the full-token `Qwen/Qwen3-VL-2B-Instruct` baseline
used by the project.

- Editable source: `qwen3-vl-2b-instruct-baseline.drawio`
- Image exports: `exports/`
- Hugging Face snapshot inspected:
  `89644892e4d85e24eaac8bacfd4f463576704203`
- Local Transformers implementation inspected: `5.19.0`
- Runtime configuration: `configs/baseline_qwen3_vl.yaml`

The Draw.io file contains five pages:

1. **Clean overview** — a low-crossing, left-to-right view of input processing,
   both encoder paths, multimodal fusion, decoding, and evaluation.
2. **Vision encoder detail** — patch embedding, position encoding, all 24
   Vision Transformer blocks, DeepStack taps, the internal VisionBlock, and
   the 2×2 PatchMerger.
3. **Fusion + decoder overview** — masked-scatter fusion, M-RoPE, all 28
   decoder layers, three DeepStack additions, grouped-query attention, SwiGLU,
   tied LM head, and generation.
4. **Token flow and fusion** — expansion of image placeholders, text and image
   embedding paths, exact visual-token counting, in-place replacement, the
   resulting mixed token sequence, and multimodal position IDs.
5. **Decoder architecture** — the complete 28-layer path, DeepStack injection
   after decoder layers 0/1/2, the internal attention and SwiGLU sublayers,
   residual connections, KV cache, and autoregressive generation.

The baseline retains every visual token produced by the built-in spatial patch
merger. The built-in merger is documented separately from future experimental
token pruning to avoid conflating the two mechanisms.

## Token flow and accounting

The image processor reports a grid `(Tgrid, Hgrid, Wgrid)`. With the checkpoint's
spatial merge size of 2, the number of visual embeddings is:

```text
Nvisual = (Tgrid × Hgrid × Wgrid) / 2²
```

Before tokenization finishes, the processor expands the image marker to exactly
`Nvisual` `<|image_pad|>` placeholders. The tokenizer therefore creates one
sequence containing ordinary text tokens, chat/control tokens, vision boundary
tokens, and image-placeholder positions. The language embedding table first
maps every position to a 2048-dimensional vector. The model then creates
`image_mask = (input_ids == image_token_id)`, checks that the number of mask
positions equals `Nvisual`, and uses `masked_scatter` to replace those placeholder
vectors in-place with the vision encoder's `[Nvisual, 2048]` output.

The decoder input length is consequently:

```text
Ntotal = Ntext_and_special + Nvisual
```

The baseline's reported `input_tokens` is `input_ids.shape[-1]`, i.e. `Ntotal`.
DeepStack does not change this count: its three tensors are residual features
added to the already-existing visual positions after decoder layers 0, 1, and 2.

Multimodal `position_ids` have shape `[3, batch, Ntotal]`. The three rows are
temporal, height, and width coordinates. Text positions use a sequential index
repeated across all three rows; visual positions use coordinates on the merged
visual grid. Qwen3-VL interleaves these dimensions in M-RoPE with sections
`[24, 20, 20]`.

## Decoder block

All text and visual positions enter the same decoder-only Transformer. Each of
the 28 pre-norm layers applies:

1. RMSNorm.
2. Bias-free Q/K/V projections: 16 query heads and 8 key/value heads, each with
   head dimension 128. Q and K receive per-head RMSNorm and M-RoPE.
3. Causal grouped-query attention, a 2048-to-2048 output projection, and a
   residual addition.
4. RMSNorm followed by a bias-free SwiGLU MLP:
   `SiLU(gate_proj(x)) * up_proj(x)`, with width `2048 → 6144 → 2048`, then a
   second residual addition.

After layer 27, final RMSNorm and the tied language-model head produce 151,936
vocabulary logits. Prefill processes all text and visual positions together.
During autoregressive decoding, each step appends one new text token while the
image tokens stay fixed; the KV cache avoids recomputing earlier key/value
states.

## Verified DeepStack behavior

This exact 2B checkpoint uses a 24-layer Vision Transformer with hidden size
1024 and fixed `deepstack_visual_indexes: [5, 11, 17]`. The indexes are
zero-based, so the tapped hidden states are taken after the 6th, 12th, and 18th
ViT layers. This differs from generic diagrams based on the default/larger
27-layer vision configuration with indexes `[8, 16, 24]`.

The indexes are checkpoint configuration, not a dynamic per-image layer
selection policy. Each tapped hidden state passes through its own 2×2
DeepStack PatchMerger and 2048-dimensional projection. The three resulting
feature streams are added only at visual-token positions after language
decoder layers 0, 1, and 2. In parallel, the main visual stream still executes
all 24 ViT layers and the final PatchMerger. Therefore, DeepStack is multi-level
feature reuse and must not be described as token pruning.

Verification sources:

- [Qwen3-VL-2B-Instruct checkpoint config](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json)
- [Hugging Face Transformers Qwen3-VL implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
