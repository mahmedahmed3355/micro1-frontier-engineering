from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrectnessResult:
    passed: bool
    max_abs_error: float
    max_rel_error: float
    rtol: float
    atol: float
    reason: str


def check_correctness(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-3,
) -> CorrectnessResult:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)

    if reference.shape != candidate.shape:
        return CorrectnessResult(
            passed=False,
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            rtol=rtol,
            atol=atol,
            reason=(f"Shape mismatch: {reference.shape} != {candidate.shape}"),
        )

    if not np.all(np.isfinite(candidate)):
        return CorrectnessResult(
            passed=False,
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            rtol=rtol,
            atol=atol,
            reason="Candidate contains NaN or infinity.",
        )

    if not np.all(np.isfinite(reference)):
        return CorrectnessResult(
            passed=False,
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            rtol=rtol,
            atol=atol,
            reason="Reference contains NaN or infinity.",
        )

    difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))

    denominator = np.maximum(
        np.abs(reference.astype(np.float64)),
        np.finfo(np.float64).tiny,
    )

    relative_error = difference / denominator

    max_abs_error = float(np.max(difference))
    max_rel_error = float(np.max(relative_error))

    passed = bool(
        np.allclose(
            reference,
            candidate,
            rtol=rtol,
            atol=atol,
        )
    )

    return CorrectnessResult(
        passed=passed,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        rtol=rtol,
        atol=atol,
        reason=(
            "Outputs are within tolerance." if passed else "Outputs exceed tolerance."
        ),
    )
