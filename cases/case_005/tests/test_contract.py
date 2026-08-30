from __future__ import annotations

import re
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CASE_ROOT / "src" / "lifetime.cu"
REFERENCE = CASE_ROOT / "reference" / "lifetime.cu"


def test_source_exists() -> None:
    assert SOURCE.is_file()


def test_reference_exists() -> None:
    assert REFERENCE.is_file()


def test_source_allocates_both_device_buffers() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "cudaMalloc" in text
    assert "device_input" in text
    assert "device_output" in text


def test_reference_allocates_both_device_buffers() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert text.count("cudaMalloc") == 2
    assert "device_input" in text
    assert "device_output" in text


def test_source_contains_host_device_transfers() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "cudaMemcpyHostToDevice" in text
    assert "cudaMemcpyDeviceToHost" in text


def test_source_has_output_cleanup_on_failure_path() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    d2h_position = text.find("cudaMemcpyDeviceToHost")
    success_cleanup_start = text.find(
        "cudaError_t input_free_status = cudaFree(device_input);"
    )

    assert d2h_position >= 0
    assert success_cleanup_start > d2h_position

    prefix = text[:success_cleanup_start]
    assert "cudaFree(device_output)" in prefix


def test_source_success_path_omits_output_cleanup() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    input_free_position = text.find(
        "cudaError_t input_free_status = cudaFree(device_input);"
    )
    success_return = text.rfind("return static_cast<int>(input_free_status);")

    assert input_free_position >= 0
    assert success_return > input_free_position

    success_cleanup = text[input_free_position:success_return]

    assert "cudaError_t output_free_status" not in success_cleanup
    assert not re.search(
        r"cudaFree\s*\(\s*device_output\s*\)",
        success_cleanup,
    )


def test_reference_success_path_releases_output_before_input() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    d2h_position = text.find("cudaMemcpyDeviceToHost")
    output_free_position = text.find(
        "cudaError_t output_free_status = cudaFree(device_output);"
    )
    input_free_position = text.find(
        "cudaError_t input_free_status = cudaFree(device_input);"
    )
    return_position = text.rfind("return static_cast<int>(input_free_status);")

    assert d2h_position >= 0
    assert output_free_position > d2h_position
    assert input_free_position > output_free_position
    assert return_position > input_free_position


def test_reference_checks_output_cleanup_status() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    output_free_position = text.find(
        "cudaError_t output_free_status = cudaFree(device_output);"
    )
    output_check_position = text.find("if (output_free_status != cudaSuccess)")

    assert output_free_position >= 0
    assert output_check_position > output_free_position


def test_reference_releases_input_after_output() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    output_free = text.find("cudaError_t output_free_status = cudaFree(device_output);")
    input_free = text.find("cudaError_t input_free_status = cudaFree(device_input);")

    assert output_free >= 0
    assert input_free > output_free
