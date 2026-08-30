from __future__ import annotations

from typing import Any

from agents.base import LLMClient


class GeminiFlashClient(LLMClient):
    """
    Minimal Gemini Flash adapter implementing the project's LLMClient contract.

    The Gemini SDK is isolated here so agents remain provider-agnostic.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.6-flash",
    ) -> None:
        if not api_key:
            raise ValueError("Gemini API key is required.")

        from google import genai

        self.model = model
        self.client = genai.Client(
            api_key=api_key
        )

    def invoke(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        del metadata

        contents = prompt

        if system_prompt:
            contents = (
                f"{system_prompt}\n\n"
                f"{prompt}"
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
        )

        text = getattr(
            response,
            "text",
            None,
        )

        if text is None:
            raise RuntimeError(
                "Gemini returned no text response."
            )

        return text
