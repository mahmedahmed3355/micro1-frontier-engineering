from orchestration.graph import build_graph


class FakeOptimizer:
    def __init__(self):
        self.calls = []

    def __call__(self, state):
        self.calls.append(
            {
                "iteration": state["iteration"],
                "feedback": state.get("latest_feedback", {}),
                "rejected_hashes": state.get("rejected_candidate_hashes", []),
            }
        )
        iteration = state["iteration"]
        return {
            "candidate_source": f"candidate-{iteration}",
            "candidate_sha256": f"hash-{iteration}",
        }


class FakeVerifier:
    def __init__(self):
        self.calls = []

    def __call__(self, state):
        self.calls.append(state["candidate_source"])

        if len(self.calls) == 1:
            return {
                "accepted": False,
                "latest_feedback": {
                    "summary": "Candidate did not improve performance.",
                },
            }

        return {"accepted": True, "status": "accepted"}


def test_graph_retries_optimizer_after_verifier_rejection():
    optimizer = FakeOptimizer()
    verifier = FakeVerifier()

    graph = build_graph(
        reader=lambda state: {"code_map": {}},
        analyzer=lambda state: {"performance_analysis": {}},
        optimizer=optimizer,
        verifier=verifier,
    )

    result = graph.invoke(
        {
            "case_id": "case_001",
            "iteration": 0,
            "max_iterations": 3,
            "rejected_candidate_hashes": [],
        }
    )

    assert result["accepted"] is True

    assert len(optimizer.calls) == 2

    assert optimizer.calls[0]["feedback"] == {}

    assert "did not improve" in optimizer.calls[1]["feedback"]["summary"]
    assert optimizer.calls[1]["rejected_hashes"] == ["hash-0"]

    assert result["candidate_source"] == "candidate-1"


def test_retry_deduplicates_rejected_candidate_hashes():
    observed_hash_lists = []

    def optimizer(state):
        observed_hash_lists.append(list(state.get("rejected_candidate_hashes", [])))
        return {
            "candidate_source": "same candidate",
            "candidate_sha256": "same-hash",
        }

    calls = {"count": 0}

    def verifier(state):
        calls["count"] += 1
        return {
            "accepted": calls["count"] == 3,
            "latest_feedback": {"summary": "retry"},
        }

    graph = build_graph(
        reader=lambda state: {"code_map": {}},
        analyzer=lambda state: {"performance_analysis": {}},
        optimizer=optimizer,
        verifier=verifier,
    )

    graph.invoke(
        {
            "iteration": 0,
            "max_iterations": 3,
            "rejected_candidate_hashes": [],
        }
    )

    assert observed_hash_lists == [[], ["same-hash"], ["same-hash"]]


def test_final_rejection_records_its_hash_before_exhausting():
    graph = build_graph(
        reader=lambda state: {"code_map": {}},
        analyzer=lambda state: {"performance_analysis": {}},
        optimizer=lambda state: {
            "candidate_source": "candidate",
            "candidate_sha256": "final-hash",
        },
        verifier=lambda state: {"accepted": False},
    )

    result = graph.invoke(
        {
            "iteration": 0,
            "max_iterations": 1,
            "rejected_candidate_hashes": [],
        }
    )

    assert result["status"] == "exhausted"
    assert result["rejected_candidate_hashes"] == ["final-hash"]
