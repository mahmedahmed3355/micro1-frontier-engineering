from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.orchestrator.graph import build_graph
from runtime.result import CasePaths


def load_case(
    case_dir: str | Path,
) -> tuple[
    CasePaths,
    dict[str, Any],
]:
    root = Path(case_dir)

    if not root.exists():
        raise FileNotFoundError(f"Case directory does not exist: {root}")

    paths = CasePaths(root)

    required = [
        paths.input,
        paths.reference,
        paths.metadata,
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing case files:\n" + "\n".join(str(path) for path in missing)
        )

    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))

    evidence_path = root / "baseline_evidence.json"
    baseline_evidence = (
        json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence_path.exists()
        else {}
    )

    return paths, {**metadata, "baseline_evidence": baseline_evidence}


def run_case(
    case_dir: str | Path,
    *,
    reader: Any,
    analyzer: Any,
    optimizer: Any,
    verifier: Any,
    initial_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths, case_data = load_case(case_dir)

    source = paths.input.read_text(encoding="utf-8")

    resume = initial_state is not None

    graph = build_graph(
        reader=reader.run if hasattr(reader, "run") else reader,
        analyzer=analyzer.run if hasattr(analyzer, "run") else analyzer,
        optimizer=optimizer.run if hasattr(optimizer, "run") else optimizer,
        verifier=verifier.run if hasattr(verifier, "run") else verifier,
        start_from="optimizer" if resume else "reader",
    )

    default_state: dict[str, Any] = {
        "case_id": case_data.get(
            "case_id",
            paths.root.name,
        ),
        "case_dir": str(paths.root),
        "workspace": str(paths.root),
        "input_files": [
            "input.cu",
            "reference.cu",
            "metadata.json",
        ],
        "input_source": source,
        "metadata": {
            key: value for key, value in case_data.items() if key != "baseline_evidence"
        },
        "baseline_evidence": case_data["baseline_evidence"],
        "iteration": 0,
        "max_iterations": 3,
        "feedback": [],
        "feedback_history": [],
        "rejected_candidate_hashes": [],
        "optimization_history": [],
        "status": "running",
    }

    if initial_state is not None:
        state_input = {
            **default_state,
            **initial_state,
        }
    else:
        state_input = default_state

    state = graph.invoke(state_input)

    candidate_source = state.get(
        "candidate_source",
        "",
    )

    if candidate_source and state.get("accepted", False):
        paths.candidate.write_text(
            candidate_source,
            encoding="utf-8",
        )

    summary = {
        "case_id": state.get("case_id"),
        "iteration": state.get(
            "iteration",
            0,
        ),
        "accepted": state.get(
            "accepted",
            False,
        ),
        "feedback": state.get(
            "feedback",
            "",
        ),
        "has_candidate": bool(candidate_source),
        "status": state.get("status"),
        "candidate_sha256": state.get("candidate_sha256"),
    }

    paths.summary.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    return summary
