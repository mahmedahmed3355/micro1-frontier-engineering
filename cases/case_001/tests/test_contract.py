from __future__ import annotations

import re
from pathlib import Path

CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = CASE_ROOT / "src" / "kernel.cu"
REFERENCE = CASE_ROOT / "reference" / "kernel.cu"


def test_defective_source_exists() -> None:
    assert SOURCE.is_file()


def test_reference_source_exists() -> None:
    assert REFERENCE.is_file()


def test_source_contains_kernel_and_launcher() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "__global__ void vector_add_kernel" in text
    assert "launch_vector_add" in text


def test_reference_contains_bounds_guard() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert re.search(
        r"if\s*\(\s*idx\s*<\s*n\s*\)",
        text,
    )


def test_defective_source_exposes_boundary_bug() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "const int idx = blockIdx.x * blockDim.x + threadIdx.x;" in text

    # The defective implementation must not already contain the
    # required logical-index bounds guard.
    assert not re.search(
        r"if\s*\(\s*idx\s*<\s*n\s*\)",
        text,
    )


def test_launch_rounds_grid_up() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    assert "(n + block_size - 1) / block_size" in text


def test_reference_preserves_rounded_grid() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    assert "(n + block_size - 1) / block_size" in text
