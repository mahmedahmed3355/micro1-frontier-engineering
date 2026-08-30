from __future__ import annotations

import re
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CASE_ROOT / "src" / "training.py"
REFERENCE = CASE_ROOT / "reference" / "training.py"


def test_source_exists() -> None:
    assert SOURCE.is_file()


def test_reference_exists() -> None:
    assert REFERENCE.is_file()


def test_source_contains_distributed_guard() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "dist.is_available()" in text
    assert "dist.is_initialized()" in text


def test_reference_contains_distributed_guard() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert "dist.is_available()" in text
    assert "dist.is_initialized()" in text


def test_source_has_backward_before_optimizer_step() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    backward = text.find("loss.backward()")
    optimizer = text.find("optimizer.step()")

    assert backward >= 0
    assert optimizer > backward


def test_reference_synchronizes_before_optimizer_step() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    barrier = text.find("dist.barrier()")
    optimizer = text.find("optimizer.step()")

    assert barrier >= 0
    assert optimizer > barrier


def test_reference_aggregates_loss_before_optimizer_step() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    reduce_match = re.search(r"dist\.all_reduce\s*\(", text)
    optimizer = text.find("optimizer.step()")

    assert reduce_match is not None
    assert optimizer > reduce_match.start()


def test_source_preserves_synchronization_defect() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "dist.barrier()" in text
    assert "dist.all_reduce(" not in text


def test_reference_contains_world_size_normalization() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert "dist.get_world_size()" in text
