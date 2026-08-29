from pathlib import Path

import numpy as np
import pytest

from tools.benchmark import benchmark_callable, percentile
from tools.correctness import check_correctness
from tools.filesystem import (
    list_files,
    read_text_file,
    write_text_file,
)


def test_filesystem_round_trip(tmp_path: Path):
    target = tmp_path / "nested" / "sample.txt"

    write_text_file(
        target,
        "hello cuda",
    )

    assert read_text_file(target) == "hello cuda"
    assert target in list_files(
        tmp_path,
        "*.txt",
    )


def test_correctness_accepts_close_arrays():
    reference = np.asarray(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    candidate = np.asarray(
        [1.0, 2.0, 3.000001],
        dtype=np.float32,
    )

    result = check_correctness(
        reference,
        candidate,
    )

    assert result.passed is True
    assert result.max_abs_error > 0.0


def test_correctness_rejects_large_difference():
    reference = np.asarray(
        [1.0, 2.0, 3.0],
        dtype=np.float32,
    )

    candidate = np.asarray(
        [1.0, 2.0, 9.0],
        dtype=np.float32,
    )

    result = check_correctness(
        reference,
        candidate,
    )

    assert result.passed is False


def test_correctness_rejects_shape_mismatch():
    result = check_correctness(
        np.asarray([1.0, 2.0]),
        np.asarray([1.0]),
    )

    assert result.passed is False
    assert "Shape mismatch" in result.reason


def test_correctness_rejects_nan():
    result = check_correctness(
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, np.nan]),
    )

    assert result.passed is False


def test_percentile_empty_values():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_percentile_invalid_value():
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 101)


def test_benchmark_callable():
    counter = {"value": 0}

    def function():
        counter["value"] += 1

    result = benchmark_callable(
        function,
        runs=5,
        warmup_runs=2,
    )

    assert result.runs == 5
    assert result.warmup_runs == 2
    assert len(result.samples_ms) == 5
    assert result.median_ms >= 0.0
    assert result.p95_ms >= result.median_ms
    assert counter["value"] == 7
