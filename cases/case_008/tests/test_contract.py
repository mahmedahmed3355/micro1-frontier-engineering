from __future__ import annotations

import re
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CASE_ROOT / "src" / "optimizer.py"
REFERENCE = CASE_ROOT / "reference" / "optimizer.py"


def test_source_exists() -> None:
    assert SOURCE.is_file()


def test_reference_exists() -> None:
    assert REFERENCE.is_file()


def test_source_has_distributed_guard() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "dist.is_available()" in text
    assert "dist.is_initialized()" in text


def test_reference_has_distributed_guard() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert "dist.is_available()" in text
    assert "dist.is_initialized()" in text


def test_source_communicates_gradient() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "dist.all_reduce" in text
    assert "local_gradient" in text


def test_reference_communicates_gradient() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert "dist.all_reduce" in text
    assert "local_gradient" in text


def test_reference_normalizes_before_update() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    normalize = text.find("local_gradient.div_(dist.get_world_size())")
    update = text.find("parameter.add_(local_gradient")

    assert normalize >= 0
    assert update > normalize


def test_source_preserves_missing_normalization_defect() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "dist.all_reduce(local_gradient" in text
    assert "local_gradient.div_(dist.get_world_size())" not in text


def test_reference_uses_world_size_normalization() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert re.search(
        r"local_gradient\.div_\s*\(\s*dist\.get_world_size\s*\(\s*\)\s*\)",
        text,
    )


def test_update_occurs_after_gradient_communication() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    reduce_position = text.find("dist.all_reduce(local_gradient")
    update_position = text.find("parameter.add_(local_gradient")

    assert reduce_position >= 0
    assert update_position > reduce_position
