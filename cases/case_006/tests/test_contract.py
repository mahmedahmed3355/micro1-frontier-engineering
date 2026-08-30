from __future__ import annotations

import re
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CASE_ROOT / "src" / "device_mismatch.py"
REFERENCE = CASE_ROOT / "reference" / "device_mismatch.py"


def test_source_has_expected_entrypoint() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "def run_device_mismatch(" in text
    assert "values: torch.Tensor" in text
    assert "weight: torch.Tensor" in text


def test_source_imports_torch() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "import torch" in text
    assert "torch.device" in text


def test_source_moves_values_to_execution_device() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert re.search(
        r"values\s*=\s*values\.to\s*\(\s*device\s*\)",
        text,
    )


def test_source_uses_weight_in_computation() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "values * weight" in text


def test_defective_source_does_not_move_weight_to_device() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    weight_device_moves = re.findall(
        r"weight\s*=\s*weight\.to\s*\(\s*device\s*\)",
        text,
    )

    assert not weight_device_moves


def test_reference_moves_values_and_weight_to_same_device() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert re.search(
        r"values\s*=\s*values\.to\s*\(\s*device\s*\)",
        text,
    )
    assert re.search(
        r"weight\s*=\s*weight\.to\s*\(\s*device\s*\)",
        text,
    )


def test_reference_performs_computation_after_device_normalization() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    values_position = text.find("values = values.to(device)")
    weight_position = text.find("weight = weight.to(device)")
    operation_position = text.find("return values * weight")

    assert values_position >= 0
    assert weight_position > values_position
    assert operation_position > weight_position


def test_reference_contains_no_second_unrelated_device_path() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert text.count("device = torch.device") == 1
