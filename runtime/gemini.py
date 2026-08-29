from __future__ import annotations

from typing import Any

from google import genai
from kaggle_secrets import UserSecretsClient

from agents.base import LLMClient


class GeminiClient(LLMClient):
    """
    Gemini API adapter for the project's LLMClient interface.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-3.6-flash",
        api_key: str | None = None,
        max_output_tokens: int = 768,
    ) -> None:
        if api_key is None:
            api_key = UserSecretsClient().get_secret("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not available.")

        self.model = model
        self.max_output_tokens = max_output_tokens
        self.client = genai.Client(api_key=api_key)

    def invoke(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        config: dict[str, Any] = {
            "max_output_tokens": self.max_output_tokens,
        }

        if system_prompt:
            config["system_instruction"] = system_prompt

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        text = response.text

        if text is None:
            raise RuntimeError("Gemini returned an empty response.")

        return text.strip()
