from agents.verifier import VerifierAgent
from runtime.verifier import _output_contract_errors


class FakeDeterministicVerifier:
    def __init__(self):
        self.states = []

    def verify(self, state):
        self.states.append(state)
        return {
            "decision": "REJECTED",
            "compilation": {"success": True},
            "correctness": {"passed": False},
            "performance": {"qualifying": False},
            "feedback": {"summary": "Numerical mismatch."},
        }


def test_verifier_uses_deterministic_result_without_an_llm():
    deterministic = FakeDeterministicVerifier()
    agent = VerifierAgent(verifier=deterministic)

    result = agent.run(
        {
            "feedback": [],
        }
    )

    assert result["verification"]["decision"] == "REJECTED"
    assert result["correctness_result"]["passed"] is False
    assert result["latest_feedback"]["summary"] == "Numerical mismatch."
    assert result["feedback_history"] == [{"summary": "Numerical mismatch."}]
    assert deterministic.states


def test_output_contract_rejects_size_mismatch():
    errors = _output_contract_errors(
        {
            "SIZE": "256",
            "KERNEL_TIME_MS": "0.5",
            "RESULT_SAMPLE": ",".join(["1.0"] * 10),
        },
        512,
    )

    assert errors == ["SIZE must equal 512, got 256."]


def test_output_contract_rejects_non_numeric_kernel_time():
    errors = _output_contract_errors(
        {
            "SIZE": "512",
            "KERNEL_TIME_MS": "not-a-number",
            "RESULT_SAMPLE": ",".join(["1.0"] * 10),
        },
        512,
    )

    assert errors == ["KERNEL_TIME_MS must be numeric."]


def test_output_contract_rejects_wrong_result_sample_length():
    errors = _output_contract_errors(
        {
            "SIZE": "512",
            "KERNEL_TIME_MS": "0.5",
            "RESULT_SAMPLE": "1.0,2.0,3.0",
        },
        512,
    )

    assert errors == ["RESULT_SAMPLE must contain exactly 10 values."]
