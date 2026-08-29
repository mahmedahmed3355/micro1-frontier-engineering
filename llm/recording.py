from __future__ import annotations

from typing import Any

from agents.base import LLMClient
from trajectories.recorder import TrajectoryRecorder


class RecordingLLM(LLMClient):
    """
    Wrap any LLM client and record every interaction.
    """

    def __init__(
        self,
        client: LLMClient,
        recorder: TrajectoryRecorder,
    ) -> None:
        self.client = client
        self.recorder = recorder

    def invoke(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        response = self.client.invoke(
            prompt,
            system_prompt=system_prompt,
            metadata=metadata,
        )

        self.recorder.record(
            agent=str(
                (metadata or {}).get(
                    "agent",
                    "unknown",
                )
            ),
            prompt=prompt,
            response=response,
            metadata={
                **(metadata or {}),
                "has_system_prompt": (system_prompt is not None),
            },
        )

        return response
