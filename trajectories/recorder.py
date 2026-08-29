from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TrajectoryRecorder:
    """
    Append-only JSONL trajectory recorder.

    Every LLM interaction is recorded as one event.

    Secrets such as API keys are never written to the trajectory.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def record(
        self,
        *,
        agent: str,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "agent": agent,
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {},
        }

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                )
                + "\n"
            )

    def record_verification(
        self,
        verification: dict[str, Any],
    ) -> None:
        """Record deterministic verifier evidence in the same JSONL log."""
        self.record(
            agent="deterministic_verifier",
            prompt="",
            response="",
            metadata={
                "event_type": "verification",
                "verification": verification,
            },
        )
