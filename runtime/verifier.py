from __future__ import annotations

import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.compiler import compile_cuda
from tools.correctness import check_correctness
from tools.runner import run_executable

_FIELD = re.compile(r"^(KERNEL_TIME_MS|SIZE|RESULT_SAMPLE)=(.*)$")


def _parse_output(output: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    for line in output.splitlines():
        match = _FIELD.match(line.strip())
        if match:
            fields[match.group(1)] = match.group(2)
    missing = [
        key for key in ("KERNEL_TIME_MS", "SIZE", "RESULT_SAMPLE") if key not in fields
    ]
    return fields, missing


def _sample(fields: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(value) for value in fields["RESULT_SAMPLE"].split(",")],
        dtype=np.float32,
    )


def _output_contract_errors(
    fields: dict[str, str],
    expected_size: int,
) -> list[str]:
    errors: list[str] = []
    try:
        if int(fields["SIZE"]) != expected_size:
            errors.append(f"SIZE must equal {expected_size}, got {fields['SIZE']}.")
    except (KeyError, ValueError):
        errors.append("SIZE must be an integer.")

    try:
        kernel_time = float(fields["KERNEL_TIME_MS"])
        if not math.isfinite(kernel_time):
            errors.append("KERNEL_TIME_MS must be finite.")
    except (KeyError, ValueError):
        errors.append("KERNEL_TIME_MS must be numeric.")

    try:
        values = [float(value) for value in fields["RESULT_SAMPLE"].split(",")]
        if len(values) != 10:
            errors.append("RESULT_SAMPLE must contain exactly 10 values.")
        elif not all(math.isfinite(value) for value in values):
            errors.append("RESULT_SAMPLE values must be finite.")
    except (KeyError, ValueError):
        errors.append("RESULT_SAMPLE must contain numeric values.")

    return errors


class DeterministicVerifier:
    """Compile and evaluate exactly the generated candidate source."""

    def verify(self, state: dict[str, Any]) -> dict[str, Any]:
        case_dir = Path(state["case_dir"])
        metadata = state["metadata"]
        iteration = int(state.get("iteration", 0))
        candidate_source = state.get("candidate_source", "")
        candidate_sha256 = state.get("candidate_sha256", "")
        artifacts = case_dir / ".agent_artifacts" / "candidates"
        source_path = artifacts / f"candidate-{iteration}-{candidate_sha256[:12]}.cu"
        executable = artifacts / f"candidate-{iteration}-{candidate_sha256[:12]}"

        candidate = {
            "iteration": iteration,
            "sha256": candidate_sha256,
            "source_path": str(source_path),
        }
        if not candidate_source:
            return self._failure(
                candidate,
                "GENERATION_FAILURE",
                "Candidate source was not a single fenced CUDA block.",
            )

        artifacts.mkdir(parents=True, exist_ok=True)
        source_path.write_text(candidate_source, encoding="utf-8")
        compilation = compile_cuda(source_path, executable)
        compilation_data = asdict(compilation)
        compilation_data["executable"] = str(compilation.executable)
        if not compilation.success:
            return self._failure(
                candidate,
                "COMPILATION_FAILURE",
                "Candidate did not compile.",
                compilation=compilation_data,
            )

        size = str(metadata["default_size"])
        candidate_run = run_executable(executable, [size])
        execution = asdict(candidate_run)
        if not candidate_run.success:
            return self._failure(
                candidate,
                "EXECUTION_FAILURE",
                "Candidate executable failed.",
                compilation=compilation_data,
                execution=execution,
            )

        expected_size = int(metadata["default_size"])
        candidate_fields, missing = _parse_output(candidate_run.stdout)
        contract_errors = (
            _output_contract_errors(
                candidate_fields,
                expected_size,
            )
            if not missing
            else []
        )
        if missing or contract_errors:
            return self._failure(
                candidate,
                "OUTPUT_CONTRACT_FAILURE",
                "Candidate violated the benchmark output contract.",
                compilation=compilation_data,
                execution=execution,
                output_contract={
                    "passed": False,
                    "missing_fields": missing,
                    "errors": contract_errors,
                },
            )

        reference = self._reference_fields(case_dir, artifacts, size)
        if isinstance(reference, dict) and "failure_stage" in reference:
            return self._failure(
                candidate,
                reference["failure_stage"],
                reference["summary"],
                compilation=compilation_data,
                execution=execution,
            )
        reference_errors: list[str] = []

        try:
            if int(reference["SIZE"]) != expected_size:
                reference_errors.append(
                    f"SIZE must equal {expected_size}, got {reference['SIZE']}."
                )
        except (KeyError, ValueError):
            reference_errors.append("SIZE must be an integer.")

        try:
            reference_values = [
                float(value) for value in reference["RESULT_SAMPLE"].split(",")
            ]
            if len(reference_values) != 10:
                reference_errors.append("RESULT_SAMPLE must contain exactly 10 values.")
            elif not all(math.isfinite(value) for value in reference_values):
                reference_errors.append("RESULT_SAMPLE values must be finite.")
        except (KeyError, ValueError):
            reference_errors.append("RESULT_SAMPLE must contain numeric values.")

        if reference_errors:
            return self._failure(
                candidate,
                "REFERENCE_OUTPUT_CONTRACT_FAILURE",
                "Reference violated the reference output contract.",
                compilation=compilation_data,
                execution=execution,
                output_contract={
                    "passed": False,
                    "missing_fields": [],
                    "errors": reference_errors,
                },
            )

        try:
            candidate_sample = _sample(candidate_fields)
            reference_sample = _sample(reference)
            comparison = check_correctness(
                reference_sample,
                candidate_sample,
                rtol=float(metadata["correctness"]["rtol"]),
                atol=float(metadata["correctness"]["atol"]),
            )
        except (KeyError, ValueError) as exc:
            return self._failure(
                candidate,
                "OUTPUT_CONTRACT_FAILURE",
                f"RESULT_SAMPLE could not be parsed: {exc}",
                compilation=compilation_data,
                execution=execution,
            )

        correctness = asdict(comparison)
        correctness["comparison_scope"] = "declared_result_sample"
        if not comparison.passed:
            mismatch = int(
                np.flatnonzero(
                    ~np.isclose(
                        reference_sample,
                        candidate_sample,
                        rtol=comparison.rtol,
                        atol=comparison.atol,
                    )
                )[0]
            )
            correctness.update(
                {
                    "first_mismatch_index": mismatch,
                    "actual": float(candidate_sample[mismatch]),
                    "expected": float(reference_sample[mismatch]),
                }
            )
            return self._failure(
                candidate,
                "CORRECTNESS_FAILURE",
                "Candidate RESULT_SAMPLE exceeds the required tolerance.",
                compilation=compilation_data,
                execution=execution,
                output_contract={
                    "passed": True,
                    "missing_fields": [],
                    "errors": [],
                },
                correctness=correctness,
            )

        baseline = state.get("baseline_evidence", {}).get("baseline", {})
        if "kernel_time_ms" not in baseline:
            return self._failure(
                candidate,
                "BASELINE_EVIDENCE_FAILURE",
                "Correct candidate cannot be scored without baseline kernel timing.",
                compilation=compilation_data,
                execution=execution,
                output_contract={
                    "passed": True,
                    "missing_fields": [],
                    "errors": [],
                },
                correctness=correctness,
            )
        baseline_ms = float(baseline["kernel_time_ms"])
        candidate_ms = float(candidate_fields["KERNEL_TIME_MS"])
        improved = candidate_ms < baseline_ms
        performance = {
            "status": "MEASURED_AFTER_CORRECTNESS",
            "qualifying": True,
            "baseline_ms": baseline_ms,
            "candidate_ms": candidate_ms,
            "speedup_ratio": baseline_ms / candidate_ms,
            "improved": improved,
        }
        return {
            "candidate": candidate,
            "decision": "ACCEPTED" if improved else "REJECTED",
            "failure_stage": None if improved else "PERFORMANCE_FAILURE",
            "compilation": compilation_data,
            "execution": execution,
            "output_contract": {
                "passed": True,
                "missing_fields": [],
                "errors": [],
            },
            "correctness": correctness,
            "performance": performance,
            "feedback": {
                "category": "accepted" if improved else "not_faster_than_baseline",
                "summary": (
                    "Candidate is correct and faster."
                    if improved
                    else "Candidate is correct but not faster than baseline."
                ),
                "do_not_repeat_sha256": candidate_sha256,
            },
        }

    def _reference_fields(
        self,
        case_dir: Path,
        artifacts: Path,
        size: str,
    ) -> dict[str, str] | dict[str, Any]:
        executable = artifacts / "reference"
        compilation = compile_cuda(case_dir / "reference.cu", executable)
        if not compilation.success:
            return {
                "failure_stage": "REFERENCE_COMPILATION_FAILURE",
                "summary": "Reference implementation did not compile.",
            }
        execution = run_executable(executable, [size])
        if not execution.success:
            return {
                "failure_stage": "REFERENCE_EXECUTION_FAILURE",
                "summary": "Reference implementation failed.",
            }
        fields: dict[str, str] = {}

        for line in execution.stdout.splitlines():
            line = line.strip()

            if line.startswith("SIZE="):
                fields["SIZE"] = line.split("=", 1)[1]
            elif line.startswith("RESULT_SAMPLE="):
                fields["RESULT_SAMPLE"] = line.split("=", 1)[1]

        missing = [key for key in ("SIZE", "RESULT_SAMPLE") if key not in fields]

        if missing:
            return {
                "failure_stage": "REFERENCE_OUTPUT_CONTRACT_FAILURE",
                "summary": (
                    "Reference omitted required output fields: " + ", ".join(missing)
                ),
            }

        return fields

    @staticmethod
    def _failure(
        candidate: dict[str, Any],
        stage: str,
        summary: str,
        **details: Any,
    ) -> dict[str, Any]:
        return {
            "candidate": candidate,
            "decision": "REJECTED",
            "failure_stage": stage,
            "compilation": details.get("compilation", {"passed": False}),
            "execution": details.get("execution", {"passed": False}),
            "output_contract": details.get(
                "output_contract",
                {"passed": False, "missing_fields": []},
            ),
            "correctness": details.get(
                "correctness",
                {"passed": False, "status": "NOT_RUN"},
            ),
            "performance": {
                "status": "SKIPPED_UNTIL_CORRECT",
                "qualifying": False,
            },
            "feedback": {
                "category": stage.lower(),
                "summary": summary,
                "do_not_repeat_sha256": candidate.get("sha256", ""),
            },
        }
