from __future__ import annotations

import json
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]


def load_json(name: str) -> dict:
    return json.loads((CASE_ROOT / "tests" / name).read_text(encoding="utf-8"))


def test_input_matrix_is_deterministic() -> None:
    data = load_json("input_matrix.json")

    assert data["case_id"] == "case_001"
    assert data["block_size"] == 256
    assert data["seed"] == 1001
    assert data["pattern"] == "deterministic_float_sequence"

    sizes = data["sizes"]

    assert sizes == sorted(sizes)
    assert len(sizes) == len(set(sizes))
    assert 1 in sizes
    assert 255 in sizes
    assert 256 in sizes
    assert 257 in sizes
    assert 4097 in sizes


def test_expected_behavior_defines_logical_boundary() -> None:
    data = load_json("expected_behavior.json")

    assert data["case_id"] == "case_001"
    assert data["valid_index_rule"] == "0 <= idx < n"
    assert data["expected_result"] == ("out[i] = a[i] + b[i] for every valid i")
    assert data["determinism"] is True


def test_verification_matrix_covers_boundary_classes() -> None:
    data = load_json("verification_matrix.json")

    assert data["case_id"] == "case_001"

    checks = {check["name"]: check["sizes"] for check in data["checks"]}

    assert "single_element" in checks
    assert "sub_block" in checks
    assert "exact_block" in checks
    assert "one_past_block" in checks
    assert "partial_final_block" in checks

    assert data["requirements"]["candidate_matches_reference"] is True
    assert data["requirements"]["all_valid_elements_checked"] is True
    assert data["requirements"]["boundary_sizes_required"] is True
    assert data["requirements"]["repeated_execution_required"] is True


def test_candidate_and_reference_are_separate_artifacts() -> None:
    candidate = CASE_ROOT / "src" / "kernel.cu"
    reference = CASE_ROOT / "reference" / "kernel.cu"

    assert candidate.is_file()
    assert reference.is_file()
    assert candidate != reference


def test_harness_contract_exists() -> None:
    path = CASE_ROOT / "tests" / "harness_contract.md"

    assert path.is_file()

    text = path.read_text(encoding="utf-8")

    required_phrases = [
        "execute the candidate and reference independently",
        "Generate deterministic input vectors",
        "Synchronize before reading device results",
        "Compare every valid output element",
        "Never use the candidate output as the expected result",
        "boundary matrix",
    ]

    for phrase in required_phrases:
        assert phrase in text
