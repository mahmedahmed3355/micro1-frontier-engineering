from __future__ import annotations

import re
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CASE_ROOT / "src" / "reduction.cu"
REFERENCE = CASE_ROOT / "reference" / "reduction.cu"


def test_source_exists() -> None:
    assert SOURCE.is_file()


def test_reference_exists() -> None:
    assert REFERENCE.is_file()


def test_public_interface_matches() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    reference = REFERENCE.read_text(encoding="utf-8")

    signature = 'extern "C" int run_reduction('

    assert signature in source
    assert signature in reference


def test_both_use_shared_memory() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    reference = REFERENCE.read_text(encoding="utf-8")

    assert "__shared__" in source
    assert "__shared__" in reference


def test_both_contain_reduction_stages() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    reference = REFERENCE.read_text(encoding="utf-8")

    assert "stride >>= 1" in source
    assert "stride >>= 1" in reference


def test_defective_implementation_omits_block_barriers() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    barrier_calls = re.findall(r"__syncthreads\s*\(\s*\)", text)

    assert barrier_calls == []


def test_reference_contains_block_barriers() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    barrier_calls = re.findall(r"__syncthreads\s*\(\s*\)", text)

    assert len(barrier_calls) >= 2


def test_reference_synchronizes_after_shared_memory_load() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    load_position = text.find("shared_values[tid] =")
    barrier_position = text.find("__syncthreads()")
    reduction_position = text.find("for (int stride")

    assert load_position >= 0
    assert barrier_position > load_position
    assert reduction_position > barrier_position


def test_reference_synchronizes_between_reduction_stages() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    reduction_position = text.find("for (int stride")
    barrier_position = text.find("__syncthreads()", reduction_position)

    assert reduction_position >= 0
    assert barrier_position > reduction_position
