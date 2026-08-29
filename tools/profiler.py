from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileResult:
    available: bool
    tool: str
    metrics: dict[str, float]
    message: str


def profile_executable(
    executable: str,
) -> ProfileResult:
    """
    Profiler interface.

    The actual NVIDIA profiling backend will be connected after
    the deterministic compiler/runner/benchmark layer is validated
    on the target CUDA environment.
    """

    return ProfileResult(
        available=False,
        tool="not-configured",
        metrics={},
        message=(f"Profiler backend is not configured for {executable}."),
    )
