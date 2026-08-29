from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from .routing import route_after_verification
from .state import AgentState

AgentCallable = Callable[
    [dict[str, Any]],
    dict[str, Any],
]


def build_graph(
    *,
    reader: AgentCallable,
    analyzer: AgentCallable,
    optimizer: AgentCallable,
    verifier: AgentCallable,
    start_from: str = "reader",
):
    """
    Build the authoritative bounded optimization graph.

    Reader and Analyzer run once. Every generated candidate is then
    deterministically verified before the graph can retry Optimizer.
    """

    graph = StateGraph(AgentState)

    graph.add_node(
        "reader",
        reader,
    )

    graph.add_node(
        "analyzer",
        analyzer,
    )

    graph.add_node(
        "optimizer",
        optimizer,
    )

    graph.add_node("verifier", verifier)

    if start_from == "reader":
        graph.add_edge(
            START,
            "reader",
        )

    elif start_from == "optimizer":
        graph.add_edge(
            START,
            "optimizer",
        )

    else:
        raise ValueError(
            f"Unsupported graph start point: {start_from!r}. "
            "Expected 'reader' or 'optimizer'."
        )

    graph.add_edge(
        "reader",
        "analyzer",
    )

    graph.add_edge(
        "analyzer",
        "optimizer",
    )

    graph.add_edge("optimizer", "verifier")

    def retry_update(state: AgentState) -> dict[str, Any]:
        candidate_sha256 = state.get("candidate_sha256", "")
        rejected_hashes = list(state.get("rejected_candidate_hashes", []))
        if candidate_sha256 and candidate_sha256 not in rejected_hashes:
            rejected_hashes.append(candidate_sha256)
        return {
            "iteration": state.get("iteration", 0) + 1,
            "previous_candidate_source": state.get("candidate_source", ""),
            "previous_candidate_sha256": state.get("candidate_sha256", ""),
            "rejected_candidate_hashes": rejected_hashes,
            "status": "running",
        }

    def exhausted_update(state: AgentState) -> dict[str, Any]:
        candidate_sha256 = state.get("candidate_sha256", "")
        rejected_hashes = list(state.get("rejected_candidate_hashes", []))
        if candidate_sha256 and candidate_sha256 not in rejected_hashes:
            rejected_hashes.append(candidate_sha256)
        return {
            "status": "exhausted",
            "rejected_candidate_hashes": rejected_hashes,
        }

    graph.add_node("retry", retry_update)
    graph.add_node("exhausted", exhausted_update)
    graph.add_conditional_edges(
        "verifier",
        route_after_verification,
        {
            "accept": END,
            "retry": "retry",
            "exhausted": "exhausted",
        },
    )
    graph.add_edge("retry", "optimizer")
    graph.add_edge("exhausted", END)

    return graph.compile()
