from agents import MockLLM
from agents.analyzer import AnalyzerAgent
from agents.optimizer import OptimizerAgent
from agents.orchestrator.graph import build_graph
from agents.reader import ReaderAgent


def accepting_verifier(state):
    return {
        "accepted": True,
        "status": "accepted",
        "verification": {"decision": "ACCEPTED"},
    }


def test_real_reader_analyzer_optimizer_pipeline(
    tmp_path,
):
    workspace = tmp_path / "test-workspace"
    source = workspace / "input.cu"

    source.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source.write_text(
        """
__global__ void test_kernel(
    const float* A,
    float* C,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx];
    }
}
""",
        encoding="utf-8",
    )

    llm = MockLLM(response="LLM observation")

    reader = ReaderAgent(llm)
    analyzer = AnalyzerAgent(llm)
    optimizer = OptimizerAgent(llm)

    graph = build_graph(
        reader=reader.run,
        analyzer=analyzer.run,
        optimizer=optimizer.run,
        verifier=accepting_verifier,
    )

    result = graph.invoke(
        {
            "workspace": str(workspace),
            "input_files": ["input.cu"],
            "iteration": 0,
            "max_iterations": 3,
            "feedback": [],
            "optimization_history": [],
        }
    )

    assert "code_map" in result

    assert "performance_analysis" in result
    assert "optimization_hypotheses" in result

    assert result["candidate_files"] == ["candidate.cu"]

    assert len(llm.calls) == 3

    assert llm.calls[0]["metadata"]["agent"] == "reader"

    assert llm.calls[1]["metadata"]["agent"] == "analyzer"

    assert llm.calls[2]["metadata"]["agent"] == "optimizer"


def test_pipeline_preserves_agent_order():
    calls = []

    def reader(state):
        calls.append("reader")

        return {
            "code_map": {
                "kernels": ["matmul"],
            }
        }

    def analyzer(state):
        assert "code_map" in state

        calls.append("analyzer")

        return {
            "performance_analysis": {
                "bottleneck": "memory",
            },
            "optimization_hypotheses": [
                {
                    "optimization": "tiling",
                }
            ],
        }

    def optimizer(state):
        assert "performance_analysis" in state
        assert "optimization_hypotheses" in state

        calls.append("optimizer")

        return {
            "candidate_files": ["candidate.cu"],
            "candidate_source": "candidate",
            "candidate_sha256": "test",
        }

    def verifier(state):
        calls.append("verifier")
        return {
            "accepted": True,
            "status": "accepted",
            "verification": {"decision": "ACCEPTED"},
        }

    graph = build_graph(
        reader=reader,
        analyzer=analyzer,
        optimizer=optimizer,
        verifier=verifier,
    )

    result = graph.invoke(
        {
            "input_files": ["input.cu"],
            "iteration": 0,
            "max_iterations": 3,
        }
    )

    assert calls == [
        "reader",
        "analyzer",
        "optimizer",
        "verifier",
    ]

    assert result["candidate_files"] == ["candidate.cu"]
