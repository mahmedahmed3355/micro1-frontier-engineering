from agents import MockLLM
from agents.reader import ReaderAgent


def test_reader_agent_uses_llm_and_returns_code_map():
    llm = MockLLM(response="CUDA kernel detected")

    agent = ReaderAgent(llm)

    result = agent.run(
        {
            "input_files": [
                "workloads/matmul/case_001/input.cu",
            ]
        }
    )

    assert result["code_map"]["raw_response"] == ("CUDA kernel detected")

    assert len(llm.calls) == 1
    assert llm.calls[0]["metadata"]["agent"] == "reader"


def test_reader_does_not_modify_input_state():
    llm = MockLLM()
    agent = ReaderAgent(llm)

    state = {
        "input_files": ["input.cu"],
    }

    result = agent.run(state)

    assert state == {
        "input_files": ["input.cu"],
    }

    assert "code_map" in result
