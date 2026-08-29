from __future__ import annotations

from typing import Any

from agents.base import LLMClient


class LocalQwenClient(LLMClient):
    """
    Adapter for a Qwen model loaded in the current process.

    The actual model/tokenizer are injected at construction time.
    This keeps the Agents independent from Transformers.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str = "cuda",
        max_new_tokens: int = 2400,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

    def invoke(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        import torch

        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_length = encoded["input_ids"].shape[-1]

        generated_tokens = generated[
            0,
            input_length:,
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return response.strip()
