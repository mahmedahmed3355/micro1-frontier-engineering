from __future__ import annotations

from typing import Any

from .base import LLMClient


class MockLLM(LLMClient):
    """
    Deterministic LLM substitute used for architecture tests.

    It deliberately does not perform real reasoning.
    """

    def __init__(self, response: str = "MOCK_RESPONSE") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "metadata": metadata or {},
            }
        )

        return self.response
