from __future__ import annotations

import re
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CASE_ROOT / "src" / "transfer.cu"
REFERENCE = CASE_ROOT / "reference" / "transfer.cu"


def test_defective_source_exists() -> None:
    assert SOURCE.is_file()


def test_reference_source_exists() -> None:
    assert REFERENCE.is_file()


def test_both_expose_same_public_interface() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    reference = REFERENCE.read_text(encoding="utf-8")

    signature = 'extern "C" int run_transfer('

    assert signature in source
    assert signature in reference


def test_defective_path_uses_async_device_to_host_transfer() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "cudaMemcpyAsync" in text
    assert "cudaMemcpyDeviceToHost" in text


def test_reference_uses_completed_device_to_host_transfer() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert "cudaMemcpy(" in text
    assert "cudaMemcpyDeviceToHost" in text


def test_defective_source_has_no_explicit_completion_before_free() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "cudaMemcpyAsync" in text
    assert "cudaFree(device_buffer)" in text

    completion_calls = re.findall(
        r"cuda(?:StreamSynchronize|DeviceSynchronize)\s*\(",
        text,
    )

    assert completion_calls == []


def test_reference_releases_device_memory_after_transfer() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    copy_position = text.find("cudaMemcpy(")
    free_position = text.find("cudaFree(device_buffer)")

    assert copy_position >= 0
    assert free_position > copy_position


def test_case_does_not_modify_public_signature() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    reference = REFERENCE.read_text(encoding="utf-8")

    assert "const float* host_input" in source
    assert "float* host_output" in source
    assert "int n" in source

    assert "const float* host_input" in reference
    assert "float* host_output" in reference
    assert "int n" in reference
