from agents import MockLLM
from agents.optimizer import OptimizerAgent


def test_optimizer_consumes_analyzer_output():
    llm = MockLLM(response="Candidate CUDA implementation generated.")

    agent = OptimizerAgent(llm)

    result = agent.run(
        {
            "input_files": ["input.cu"],
            "code_map": {
                "kernels": ["matmul"],
            },
            "performance_analysis": {
                "bottleneck": "global_memory_access",
            },
            "optimization_hypotheses": [
                {
                    "optimization": "shared_memory_tiling",
                }
            ],
            "latest_feedback": {},
            "iteration": 0,
            "optimization_history": [],
        }
    )

    assert result["candidate_files"] == ["candidate.cu"]

    assert result["optimization_history"]

    history = result["optimization_history"][0]

    assert history["agent"] == "optimizer"
    assert history["iteration"] == 0
    assert history["response"] == "Candidate CUDA implementation generated."

    assert len(llm.calls) == 1

    call = llm.calls[0]

    assert call["metadata"]["agent"] == "optimizer"
    assert "shared_memory_tiling" in call["prompt"]
    assert "global_memory_access" in call["prompt"]


def test_optimizer_preserves_previous_history():
    llm = MockLLM(response="second attempt")

    agent = OptimizerAgent(llm)

    previous = {
        "agent": "optimizer",
        "iteration": 0,
        "response": "first attempt",
    }

    result = agent.run(
        {
            "input_files": ["input.cu"],
            "code_map": {},
            "performance_analysis": {},
            "optimization_hypotheses": [],
            "latest_feedback": {
                "summary": "candidate was not faster",
            },
            "iteration": 1,
            "optimization_history": [previous],
        }
    )

    assert len(result["optimization_history"]) == 2
    assert result["optimization_history"][0] == previous
    assert result["optimization_history"][1]["iteration"] == 1
    assert result["optimization_history"][1]["response"] == ("second attempt")


def test_optimizer_does_not_modify_input_state():
    llm = MockLLM()
    agent = OptimizerAgent(llm)

    state = {
        "input_files": ["input.cu"],
        "performance_analysis": {
            "bottleneck": "memory",
        },
        "optimization_hypotheses": [],
        "latest_feedback": {},
        "iteration": 0,
        "optimization_history": [],
    }

    original_state = dict(state)

    agent.run(state)

    assert state == original_state
