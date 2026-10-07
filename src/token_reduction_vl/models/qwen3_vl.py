"""Full-token Qwen3-VL baseline wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    peak_vram_mb: float | None


class Qwen3VLBaseline:
    """Run ordinary full-visual-token generation before pruning is added."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        *,
        dtype: str | None = "auto",
        device_map: str | dict[str, Any] = "auto",
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
        max_new_tokens: int = 32,
        do_sample: bool = False,
    ) -> None:
        model_kwargs: dict[str, Any] = {"device_map": device_map}
        if dtype:
            model_kwargs["dtype"] = dtype
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

    def predict(self, image: Any, question: str) -> GenerationResult:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
            )
        latency_seconds = perf_counter() - started

        input_ids = inputs["input_ids"]
        generated_ids_trimmed = [
            output_ids[len(input_ids[i]) :] for i, output_ids in enumerate(generated_ids)
        ]
        text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024**2)
            if torch.cuda.is_available()
            else None
        )
        return GenerationResult(
            text=text,
            latency_seconds=latency_seconds,
            input_tokens=int(input_ids.shape[-1]),
            output_tokens=len(generated_ids_trimmed[0]),
            peak_vram_mb=peak_vram_mb,
        )
