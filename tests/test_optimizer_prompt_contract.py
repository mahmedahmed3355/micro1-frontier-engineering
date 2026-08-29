import inspect

from agents.optimizer import OptimizerAgent


def test_optimizer_prompt_declares_iteration_and_rejected_hashes():
    source = inspect.getsource(OptimizerAgent.run)

    assert "CURRENT ITERATION" in source
    assert "REJECTED CANDIDATE HASHES" in source
    assert "json.dumps(rejected_candidate_hashes" in source
    assert "Treat CURRENT ITERATION as the current candidate attempt" in source
