from agents.reader.agent import ReaderAgent
from tools.workspace import WorkspaceInspector


class FakeLLM:
    def __init__(self):
        self.prompt = None
        self.system_prompt = None

    def invoke(
        self,
        prompt,
        *,
        system_prompt=None,
        metadata=None,
    ):
        self.prompt = prompt
        self.system_prompt = system_prompt

        return "grounded reader response"


def test_reader_receives_actual_workspace_contents(
    tmp_path,
):
    workspace = tmp_path / "case"
    workspace.mkdir()

    source = workspace / "input.cu"

    source.write_text(
        """
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

    C[row * N + col] = 0.0f;
}
""",
        encoding="utf-8",
    )

    llm = FakeLLM()

    reader = ReaderAgent(
        llm=llm,
        inspector=WorkspaceInspector(),
    )

    result = reader.run(
        {
            "workspace": str(workspace),
            "input_files": ["input.cu"],
        }
    )

    # Verify that the actual source reached the LLM.
    assert "__global__ void matmul_naive" in llm.prompt

    assert "const float* A" in llm.prompt
    assert "const float* B" in llm.prompt
    assert "float* C" in llm.prompt
    assert "int N" in llm.prompt

    assert "blockIdx.y" in llm.prompt

    assert "blockIdx.x" in llm.prompt

    assert result["code_map"]["raw_response"] == "grounded reader response"


def test_reader_reports_inspected_files(
    tmp_path,
):
    workspace = tmp_path / "case"
    workspace.mkdir()

    (workspace / "input.cu").write_text(
        "CUDA_SOURCE",
        encoding="utf-8",
    )

    llm = FakeLLM()

    reader = ReaderAgent(
        llm=llm,
    )

    result = reader.run(
        {
            "workspace": str(workspace),
            "input_files": ["input.cu"],
        }
    )

    assert result["code_map"]["inspected_files"] == [
        {
            "path": "input.cu",
            "size_bytes": len(b"CUDA_SOURCE"),
        }
    ]
