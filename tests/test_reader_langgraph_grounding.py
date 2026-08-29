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


def test_langgraph_passes_workspace_to_grounded_reader(
    tmp_path,
):
    workspace = tmp_path / "case_001"
    workspace.mkdir()

    source = workspace / "input.cu"

    source.write_text(
        """
#include <cuda_runtime.h>

__global__ void matmul_naive(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    int col =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) {
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < N; ++k) {
        sum +=
            A[row * N + k] *
            B[k * N + col];
    }

    C[row * N + col] = sum;
}
""",
        encoding="utf-8",
    )

    llm = MockLLM(response="grounded observation")

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
            "input_files": [
                "input.cu",
            ],
            "iteration": 0,
            "max_iterations": 3,
            "feedback": [],
            "optimization_history": [],
        }
    )

    assert "code_map" in result

    assert result["code_map"]["input_files"] == [
        "input.cu",
    ]

    assert result["code_map"]["inspected_files"] == [
        {
            "path": "input.cu",
            "size_bytes": len(source.read_bytes()),
        }
    ]

    assert len(llm.calls) == 3

    reader_prompt = llm.calls[0]["prompt"]

    assert "__global__ void matmul_naive" in reader_prompt

    assert "const float* A" in reader_prompt
    assert "const float* B" in reader_prompt
    assert "float* C" in reader_prompt
    assert "int N" in reader_prompt

    assert "blockIdx.y" in reader_prompt

    assert "cuda_runtime.h" in reader_prompt
