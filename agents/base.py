from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """
    Minimal interface between our agents and an LLM.

    Agents depend on this abstraction rather than on a specific
    model provider.
    """

    @abstractmethod
    def invoke(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError
