from __future__ import annotations

from typing import Any, Literal, TypedDict


class AgentState(TypedDict, total=False):
    # ---------------------------------------------------------
    # Input
    # ---------------------------------------------------------
    workspace: str
    case_dir: str
    case_id: str
    input_files: list[str]
    input_source: str
    metadata: dict[str, Any]
    baseline_evidence: dict[str, Any]

    # ---------------------------------------------------------
    # Reader
    # ---------------------------------------------------------
    code_map: dict[str, Any]

    # ---------------------------------------------------------
    # Understanding / Analysis
    # ---------------------------------------------------------
    understanding: dict[str, Any]
    performance_analysis: dict[str, Any]
    optimization_hypotheses: list[dict[str, Any]]

    # ---------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------
    candidate_files: list[str]
    optimization_history: list[dict[str, Any]]
    candidate_source: str
    candidate_sha256: str
    generation_error: str
    previous_candidate_source: str
    previous_candidate_sha256: str
    rejected_candidate_hashes: list[str]

    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------
    compilation_result: dict[str, Any]
    correctness_result: dict[str, Any]
    benchmark_result: dict[str, Any]
    verification_plan: dict[str, Any]
    verification: dict[str, Any]
    latest_feedback: dict[str, Any]
    feedback_history: list[dict[str, Any]]

    # ---------------------------------------------------------
    # Orchestration
    # ---------------------------------------------------------
    iteration: int
    max_iterations: int
    accepted: bool
    status: Literal["running", "accepted", "exhausted", "failed"]
    feedback: list[dict[str, Any]]

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    best_candidate: str | None
    best_score: float | None
