from agents import MockLLM
from agents.analyzer import AnalyzerAgent


def test_analyzer_consumes_reader_context():
    llm = MockLLM(response="Global memory access may be the bottleneck.")

    agent = AnalyzerAgent(llm)

    result = agent.run(
        {
            "input_files": ["input.cu"],
            "code_map": {
                "kernels": ["matmul"],
                "memory_access": "global",
            },
        }
    )

    assert (
        result["performance_analysis"]["raw_response"]
        == "Global memory access may be the bottleneck."
    )

    assert result["optimization_hypotheses"]

    assert len(llm.calls) == 1

    call = llm.calls[0]

    assert call["metadata"]["agent"] == "analyzer"
    assert "matmul" in call["prompt"]
    assert "global" in call["prompt"].lower()


def test_analyzer_does_not_modify_state():
    llm = MockLLM()
    agent = AnalyzerAgent(llm)

    state = {
        "input_files": ["input.cu"],
        "code_map": {
            "kernels": ["matmul"],
        },
    }

    original_state = dict(state)

    result = agent.run(state)

    assert state == original_state
    assert "performance_analysis" in result
    assert "optimization_hypotheses" in result
