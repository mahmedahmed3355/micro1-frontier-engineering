from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .base import LLMClient


class BaseAgent(ABC):
    """
    Common contract for all project agents.
    """

    name: str = "base"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @abstractmethod
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Consume the current state and return state updates.
        """
        raise NotImplementedError
