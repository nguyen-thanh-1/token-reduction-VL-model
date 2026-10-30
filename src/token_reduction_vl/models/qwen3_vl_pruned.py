"""Shared inference wrapper that feeds a pruned visual-token set to Qwen3-VL.

Implements ``00-common/README.md`` §4.2. ``model.generate()`` recomputes M-RoPE
from ``input_ids`` assuming the full visual grid, so this wrapper runs its own
prefill and greedy decode and supplies ``position_ids`` explicitly:

* M-RoPE is computed once for the *unpruned* sequence, then the kept columns
  are gathered, so every kept image token keeps its original ``(t, h, w)``.
* Generated tokens get position ``max(P') + 1 + s``, identical to the baseline,
  because the last text token is never pruned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

from token_reduction_vl.pruning.base import (
    PrunerInput,
    PrunerOutput,
    VisualTokenPruner,
    validate_output,
)

INSTRUCTION_PREFIXES = ("Answer with", "Answer yes or no")


def question_text_for_scoring(prompt: str, choices: Sequence[Mapping[str, Any]] = ()) -> str:
    """Return the question plus option texts, without hint and fixed instruction.

    ``prompt`` is the output of ``build_question_prompt``: sections joined by a
    blank line, ``[Context: …] · question · [Options: …] · instruction``.
    """

    sections = [section.strip() for section in prompt.split("\n\n") if section.strip()]
    kept = [
        section
        for section in sections
        if not section.startswith(("Context:", "Options:", *INSTRUCTION_PREFIXES))
    ]
    option_texts = [str(choice.get("text", "")).strip() for choice in choices]
    return " ".join([*kept, *[text for text in option_texts if text]]).strip()


def sequence_keep_indices(seq_len: int, image_positions: Tensor, keep_indices: Tensor) -> Tensor:
    """Sequence positions to keep: every text position plus the kept image positions."""

    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=image_positions.device)
    keep_mask[image_positions] = False
    keep_mask[image_positions[keep_indices.to(image_positions.device)]] = True
    return torch.nonzero(keep_mask, as_tuple=False).flatten()


def build_position_ids(mrope: Tensor, keep_seq: Tensor) -> tuple[Tensor, int]:
    """Gather kept M-RoPE columns and prepend the plain sequence-index row.

    Returns ``([4, 1, L'], max_position)``; row 0 is the cache position used by
    the causal mask, rows 1–3 are the original ``(t, h, w)`` coordinates.
    """

    gathered = mrope[:, :, keep_seq]
    text_row = torch.arange(keep_seq.numel(), device=mrope.device, dtype=mrope.dtype).view(1, 1, -1)
    return torch.cat([text_row, gathered], dim=0), int(gathered.max().item())


def decode_position_ids(prefill_len: int, max_position: int, step: int, device: torch.device) -> Tensor:
    """Position ids ``[4, 1, 1]`` of the ``step``-th generated token (0-based)."""

    value = max_position + 1 + step
    return torch.tensor(
        [prefill_len + step, value, value, value], dtype=torch.long, device=device
    ).view(4, 1, 1)


@dataclass
class EncodedSample:
    """Processor outputs, full M-RoPE and ViT features of one sample."""

    input_ids: Tensor  # [1, L]
    mrope: Tensor  # [3, 1, L] positions of the unpruned sequence
    image_positions: Tensor  # [N] sequence indices of <|image_pad|>, ascending
    v: Tensor  # [N, 2048]
    deepstack: list[Tensor]  # 3 × [N, 2048]
    grid_hw: tuple[int, int]
    question_ids: Tensor
    question_tokens: tuple[str, ...]
    timing: dict[str, float] = field(default_factory=dict)

    @property
    def num_visual(self) -> int:
        return int(self.v.shape[0])


@dataclass
class PreparedInputs:
    inputs_embeds: Tensor  # [1, L', 2048]
    position_ids: Tensor  # [4, 1, L']
    visual_pos_masks: Tensor  # [1, L'] bool
    deepstack: list[Tensor]  # 3 × [K, 2048]
    max_position: int

    @property
    def seq_len(self) -> int:
        return int(self.inputs_embeds.shape[1])


@dataclass(frozen=True)
class PrunedGeneration:
    text: str
    output_ids: list[int]
    input_tokens: int
    output_tokens: int
    num_visual: int
    num_kept: int
    keep_indices: list[int]
    grid_hw: tuple[int, int]
    latency_seconds: float
    timing: dict[str, float]
    peak_vram_mb: float | None
    pruner_metadata: dict[str, Any]


class PrunedQwen3VL:
    """Qwen3-VL-2B with a pluggable visual-token pruner between ViT and LLM."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        *,
        dtype: str | None = "auto",
        device_map: str | dict[str, Any] = "auto",
        attn_implementation: str | None = None,
        max_new_tokens: int = 32,
    ) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_kwargs: dict[str, Any] = {"device_map": device_map}
        if dtype:
            model_kwargs["dtype"] = dtype
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        self.model.eval()
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_new_tokens = max_new_tokens
        self.image_token_id = int(self.model.config.image_token_id)
        eos = self.model.generation_config.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos])

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def embed_tokens(self) -> torch.nn.Embedding:
        return self.model.model.get_input_embeddings()

    @property
    def language_model(self) -> torch.nn.Module:
        return self.model.model.language_model

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def encode(self, image: Any, prompt: str, scoring_text: str) -> EncodedSample:
        """Run the processor and the ViT; compute M-RoPE for the full sequence."""

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        input_ids = inputs["input_ids"]
        grid_thw = inputs["image_grid_thw"]
        if grid_thw.shape[0] != 1:
            raise ValueError("The pruning wrapper supports exactly one image per sample.")
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            mm_token_type_ids = (input_ids == self.image_token_id).int()

        self._sync()
        started = perf_counter()
        vision = self.model.model.get_image_features(inputs["pixel_values"], grid_thw, return_dict=True)
        self._sync()
        vit_seconds = perf_counter() - started

        mrope, _ = self.model.model.get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=grid_thw,
            attention_mask=inputs.get("attention_mask"),
        )
        image_positions = torch.nonzero(input_ids[0] == self.image_token_id, as_tuple=False).flatten()
        v = vision.pooler_output[0]
        deepstack = [level[0] for level in vision.deepstack_features]
        merge = int(self.model.config.vision_config.spatial_merge_size)
        grid_hw = (int(grid_thw[0, 1]) // merge, int(grid_thw[0, 2]) // merge)
        if image_positions.numel() != v.shape[0] or v.shape[0] != grid_hw[0] * grid_hw[1]:
            raise ValueError("Image placeholders, visual embeddings and grid size disagree.")

        tokenizer = self.processor.tokenizer
        question_ids = tokenizer(scoring_text, add_special_tokens=False)["input_ids"]
        return EncodedSample(
            input_ids=input_ids,
            mrope=mrope,
            image_positions=image_positions,
            v=v,
            deepstack=deepstack,
            grid_hw=grid_hw,
            question_ids=torch.tensor(question_ids, dtype=torch.long, device=self.device),
            question_tokens=tuple(tokenizer.decode([token]) for token in question_ids),
            timing={"vit": vit_seconds},
        )

    def pruner_input(self, encoded: EncodedSample, ratio: float | None) -> PrunerInput:
        return PrunerInput(
            v=encoded.v,
            deepstack=encoded.deepstack,
            grid_hw=encoded.grid_hw,
            question_ids=encoded.question_ids,
            question_tokens=encoded.question_tokens,
            embed_tokens=self.embed_tokens,
            ratio=ratio,
        )

    def prepare(self, encoded: EncodedSample, selection: PrunerOutput) -> PreparedInputs:
        """Rebuild LLM inputs with only the kept visual tokens (§4.2 steps 3–4)."""

        seq_len = int(encoded.input_ids.shape[1])
        keep_seq = sequence_keep_indices(seq_len, encoded.image_positions, selection.keep_indices)
        input_ids = encoded.input_ids[:, keep_seq]
        embeds = self.embed_tokens(input_ids)
        visual_mask = input_ids == self.image_token_id
        if int(visual_mask.sum()) != selection.v_kept.shape[0]:
            raise ValueError("Kept image placeholders and kept visual embeddings disagree.")
        embeds = embeds.masked_scatter(visual_mask.unsqueeze(-1), selection.v_kept.to(embeds.dtype))
        position_ids, max_position = build_position_ids(encoded.mrope, keep_seq)
        return PreparedInputs(
            inputs_embeds=embeds,
            position_ids=position_ids,
            visual_pos_masks=visual_mask,
            deepstack=[level.to(embeds.dtype) for level in selection.deepstack_kept],
            max_position=max_position,
        )

    @torch.no_grad()
    def greedy_decode(self, prepared: PreparedInputs) -> tuple[list[int], float, float]:
        """Prefill once, then greedy decode with KV cache (§4.2 steps 5–6)."""

        lm = self.language_model
        self._sync()
        started = perf_counter()
        out = lm(
            inputs_embeds=prepared.inputs_embeds,
            position_ids=prepared.position_ids,
            attention_mask=torch.ones(prepared.inputs_embeds.shape[:2], dtype=torch.long, device=self.device),
            visual_pos_masks=prepared.visual_pos_masks,
            deepstack_visual_embeds=prepared.deepstack,
            use_cache=True,
        )
        cache = out.past_key_values
        next_id = int(self.model.lm_head(out.last_hidden_state[:, -1]).argmax(-1).item())
        self._sync()
        prefill_seconds = perf_counter() - started

        output_ids = [next_id]
        started = perf_counter()
        for step in range(1, self.max_new_tokens):
            if next_id in self.eos_token_ids:
                break
            embeds = self.embed_tokens(torch.tensor([[next_id]], device=self.device))
            out = lm(
                inputs_embeds=embeds,
                position_ids=decode_position_ids(prepared.seq_len, prepared.max_position, step - 1, self.device),
                attention_mask=torch.ones(
                    (1, prepared.seq_len + step), dtype=torch.long, device=self.device
                ),
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
            next_id = int(self.model.lm_head(out.last_hidden_state[:, -1]).argmax(-1).item())
            output_ids.append(next_id)
        self._sync()
        return output_ids, prefill_seconds, perf_counter() - started

    def predict(
        self,
        image: Any,
        prompt: str,
        scoring_text: str,
        pruner: VisualTokenPruner,
        ratio: float | None,
    ) -> PrunedGeneration:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        total_started = perf_counter()
        encoded = self.encode(image, prompt, scoring_text)
        x = self.pruner_input(encoded, ratio)
        self._sync()
        started = perf_counter()
        with torch.no_grad():
            selection = pruner.select(x)
        self._sync()
        selector_seconds = perf_counter() - started
        validate_output(x, selection)
        prepared = self.prepare(encoded, selection)
        output_ids, prefill_seconds, decode_seconds = self.greedy_decode(prepared)
        text = self.processor.batch_decode(
            [output_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return PrunedGeneration(
            text=text,
            output_ids=output_ids,
            input_tokens=prepared.seq_len,
            output_tokens=len(output_ids),
            num_visual=encoded.num_visual,
            num_kept=int(selection.keep_indices.numel()),
            keep_indices=selection.keep_indices.tolist(),
            grid_hw=encoded.grid_hw,
            latency_seconds=perf_counter() - total_started,
            timing={
                "vit": encoded.timing["vit"],
                "selector": selector_seconds,
                "prefill": prefill_seconds,
                "decode": decode_seconds,
            },
            peak_vram_mb=(
                torch.cuda.max_memory_allocated(self.device) / (1024**2)
                if self.device.type == "cuda"
                else None
            ),
            pruner_metadata=dict(selection.metadata),
        )

    def teacher_forced_logits(
        self,
        prepared: PreparedInputs,
        answer_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Logits ``[A, V]`` predicting each answer token given the (pruned) prompt.

        Gradients flow to ``prepared.inputs_embeds`` and ``prepared.deepstack``
        when the caller enables them (PP3 saliency, PP4 training). Pass a 4D
        float ``attention_mask`` ``[1, 1, L'+A, L'+A]`` to add attention biases.
        """

        answer_ids = answer_ids.to(self.device).view(1, -1)
        num_answer = int(answer_ids.shape[1])
        answer_embeds = self.embed_tokens(answer_ids).to(prepared.inputs_embeds.dtype)
        embeds = torch.cat([prepared.inputs_embeds, answer_embeds], dim=1)
        steps = torch.arange(num_answer, device=self.device)
        answer_rows = torch.stack(
            [
                prepared.seq_len + steps,
                prepared.max_position + 1 + steps,
                prepared.max_position + 1 + steps,
                prepared.max_position + 1 + steps,
            ]
        ).view(4, 1, -1)
        position_ids = torch.cat([prepared.position_ids, answer_rows.to(prepared.position_ids.dtype)], dim=2)
        visual_mask = torch.cat(
            [
                prepared.visual_pos_masks,
                torch.zeros((1, num_answer), dtype=torch.bool, device=self.device),
            ],
            dim=1,
        )
        if attention_mask is None:
            attention_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=self.device)
        out = self.language_model(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            visual_pos_masks=visual_mask,
            deepstack_visual_embeds=prepared.deepstack,
            use_cache=False,
        )
        hidden = out.last_hidden_state[0, prepared.seq_len - 1 : prepared.seq_len - 1 + num_answer]
        return self.model.lm_head(hidden).float()
