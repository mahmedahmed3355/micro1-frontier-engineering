from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tools.runner import run_executable


@dataclass(frozen=True)
class BenchmarkResult:
    samples_ms: list[float]
    median_ms: float
    p95_ms: float
    runs: int
    warmup_runs: int


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    if not values:
        raise ValueError("values cannot be empty")

    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile_value / 100.0

    lower = int(position)
    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = position - lower

    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def benchmark_callable(
    function: Callable[[], object],
    *,
    runs: int = 20,
    warmup_runs: int = 5,
) -> BenchmarkResult:
    if runs <= 0:
        raise ValueError("runs must be positive")

    if warmup_runs < 0:
        raise ValueError("warmup_runs cannot be negative")

    for _ in range(warmup_runs):
        function()

    samples: list[float] = []

    for _ in range(runs):
        start = time.perf_counter()

        function()

        elapsed = (time.perf_counter() - start) * 1000.0

        samples.append(elapsed)

    return BenchmarkResult(
        samples_ms=samples,
        median_ms=statistics.median(samples),
        p95_ms=percentile(samples, 95),
        runs=runs,
        warmup_runs=warmup_runs,
    )


def benchmark_executable(
    executable: str | Path,
    args: list[str] | None = None,
    *,
    runs: int = 20,
    warmup_runs: int = 5,
    timeout: int = 300,
) -> BenchmarkResult:
    def execute() -> None:
        result = run_executable(
            executable,
            args,
            timeout=timeout,
        )

        if not result.success:
            raise RuntimeError(f"Benchmark executable failed:\n{result.stderr}")

    return benchmark_callable(
        execute,
        runs=runs,
        warmup_runs=warmup_runs,
    )
