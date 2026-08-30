from __future__ import annotations

import re
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CASE_ROOT / "src" / "worker.py"
REFERENCE = CASE_ROOT / "reference" / "worker.py"


def test_source_exists() -> None:
    assert SOURCE.is_file()


def test_reference_exists() -> None:
    assert REFERENCE.is_file()


def test_source_creates_async_task() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "asyncio.create_task(work())" in text


def test_reference_creates_async_task() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert "asyncio.create_task(work())" in text


def test_source_awaits_task_before_return() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    task_position = text.find("asyncio.create_task(work())")
    await_position = text.find("result = await task")
    return_position = text.find("return result.cpu()")

    assert task_position >= 0
    assert await_position > task_position
    assert return_position > await_position


def test_reference_awaits_task_before_gpu_boundary() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    await_position = text.find("result = await task")
    sync_position = text.find("torch.cuda.synchronize(result.device)")

    assert await_position >= 0
    assert sync_position > await_position


def test_reference_synchronizes_before_cpu_transfer() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    sync_position = text.find("torch.cuda.synchronize(result.device)")
    cpu_position = text.find("return result.cpu()")

    assert sync_position >= 0
    assert cpu_position > sync_position


def test_source_preserves_missing_gpu_synchronization() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "result.is_cuda" not in text
    assert "torch.cuda.synchronize" not in text


def test_reference_checks_cuda_result() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert "result.is_cuda" in text


def test_reference_uses_result_device_for_synchronization() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert re.search(
        r"torch\.cuda\.synchronize\s*\(\s*result\.device\s*\)",
        text,
    )


def test_gpu_boundary_precedes_host_transfer() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    sync_position = text.find("torch.cuda.synchronize")
    transfer_position = text.find("result.cpu()")

    assert sync_position >= 0
    assert transfer_position > sync_position
