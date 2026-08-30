from __future__ import annotations

import json
from pathlib import Path

from cases._shared.contract import CASE_CONTRACTS

ROOT = Path(__file__).resolve().parents[2]


def test_all_case_contracts_exist() -> None:
    assert len(CASE_CONTRACTS) == 10

    expected_ids = {f"{index:03d}" for index in range(1, 11)}

    actual_ids = {contract.case_id for contract in CASE_CONTRACTS}

    assert actual_ids == expected_ids


def test_case_metadata_matches_contract() -> None:
    for contract in CASE_CONTRACTS:
        case_dir = ROOT / "cases" / f"case_{contract.case_id}"

        metadata_path = case_dir / "metadata.json"
        readme_path = case_dir / "README.md"

        assert metadata_path.is_file()
        assert readme_path.is_file()

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        assert metadata["case_id"] == f"case_{contract.case_id}"
        assert metadata["name"] == contract.title
        assert metadata["domain"] == contract.domain
        assert metadata["bug_class"] == contract.bug_class


def test_case_runtime_directories_exist() -> None:
    for contract in CASE_CONTRACTS:
        case_dir = ROOT / "cases" / f"case_{contract.case_id}"

        assert (case_dir / "src").is_dir()
        assert (case_dir / "reference").is_dir()
        assert (case_dir / "tests").is_dir()
        assert (case_dir / "trajectory").is_dir()
