from __future__ import annotations

import re
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CASE_ROOT / "src" / "stream_dependency.cu"
REFERENCE = CASE_ROOT / "reference" / "stream_dependency.cu"


def test_source_exists() -> None:
    assert SOURCE.is_file()


def test_reference_exists() -> None:
    assert REFERENCE.is_file()


def test_public_interface_matches() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    reference = REFERENCE.read_text(encoding="utf-8")

    signature = 'extern "C" int run_stream_dependency('

    assert signature in source
    assert signature in reference


def test_both_implementations_use_multiple_streams() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    reference = REFERENCE.read_text(encoding="utf-8")

    assert "producer_stream" in source
    assert "consumer_stream" in source
    assert "producer_stream" in reference
    assert "consumer_stream" in reference


def test_defective_implementation_omits_inter_stream_dependency() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    dependency_calls = re.findall(
        r"cuda(?:StreamWaitEvent|StreamSynchronize)\s*\(",
        text,
    )

    assert "cudaEventRecord" not in text
    assert "cudaStreamWaitEvent" not in text
    assert dependency_calls == ["cudaStreamSynchronize("]


def test_reference_contains_explicit_event_dependency() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert "cudaEventCreateWithFlags" in text
    assert "cudaEventRecord" in text
    assert "cudaStreamWaitEvent" in text


def test_reference_waits_on_consumer_before_cleanup() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    sync_marker = "status = cudaStreamSynchronize(consumer_stream)"
    sync_position = text.find(sync_marker)

    assert sync_position >= 0

    success_path = text[sync_position:]

    assert "cudaEventDestroy(producer_complete)" in success_path
    assert "cudaStreamDestroy(consumer_stream)" in success_path
    assert "cudaStreamDestroy(producer_stream)" in success_path
    assert "cudaFree(device_buffer)" in success_path


def test_kernel_is_submitted_to_consumer_stream() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    reference = REFERENCE.read_text(encoding="utf-8")

    assert "consumer_stream" in source
    assert "consumer_stream" in reference
    assert "cudaMemcpyAsync" in source
    assert "cudaMemcpyAsync" in reference
