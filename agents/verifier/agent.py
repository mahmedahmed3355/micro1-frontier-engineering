from __future__ import annotations

from typing import Any

from runtime.verifier import DeterministicVerifier
from trajectories.recorder import TrajectoryRecorder


class VerifierAgent:
    """
    Deterministic verification adapter.

    This agent does not use an LLM. Compilation, execution, output
    parsing, correctness, and performance decisions come only from
    deterministic helpers.
    """

    name = "verifier"

    def __init__(
        self,
        verifier: DeterministicVerifier | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        self.verifier = verifier or DeterministicVerifier()
        self.recorder = recorder

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        verification = self.verifier.verify(state)
        feedback = verification["feedback"]
        if self.recorder is not None:
            self.recorder.record_verification(verification)
        return {
            "verification": verification,
            "compilation_result": verification["compilation"],
            "correctness_result": verification["correctness"],
            "benchmark_result": verification["performance"],
            "latest_feedback": feedback,
            "feedback_history": [
                *state.get("feedback_history", []),
                feedback,
            ],
            "feedback": [
                *state.get("feedback", []),
                feedback,
            ],
            "accepted": verification["decision"] == "ACCEPTED",
            "status": (
                "accepted"
                if verification["decision"] == "ACCEPTED"
                else state.get("status", "running")
            ),
        }
