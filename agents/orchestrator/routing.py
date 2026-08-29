from __future__ import annotations

from .state import AgentState


def route_after_verification(state: AgentState) -> str:
    """
    Decide what the orchestrator should do after verification.

    This is deliberately deterministic.
    The LLM does not decide whether a benchmark passed.
    """

    if state.get("accepted", False):
        return "accept"

    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    if iteration + 1 >= max_iterations:
        return "exhausted"

    return "retry"
