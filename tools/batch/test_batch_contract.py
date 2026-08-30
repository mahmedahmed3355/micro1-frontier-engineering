from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "tools" / "batch" / "case_registry.json"
CONTRACT = ROOT / "tools" / "batch" / "result_contract.json"


def test_registry_contains_all_cases() -> None:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert len(data["cases"]) == 10
    assert [case["case_id"] for case in data["cases"]] == [
        f"case_{index:03d}" for index in range(1, 11)
    ]


def test_registry_paths_exist() -> None:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))

    for case in data["cases"]:
        case_path = ROOT / case["path"]

        assert case_path.is_dir()
        assert (case_path / "metadata.json").is_file()
        assert (case_path / "README.md").is_file()
        assert (case_path / "src").is_dir()
        assert (case_path / "reference").is_dir()
        assert (case_path / "tests").is_dir()
        assert (case_path / "trajectory").is_dir()


def test_batch_contract_has_required_states() -> None:
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert data["result_states"] == [
        "PASS",
        "FAIL",
        "ERROR",
        "TIMEOUT",
        "SKIPPED",
    ]


def test_external_execution_is_disabled() -> None:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert data["cuda_execution"] is False
    assert data["gemini_api"] is False
    assert data["kaggle"] is False
