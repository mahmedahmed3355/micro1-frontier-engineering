from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_ROOT = ROOT / "trajectories" / "benchmark"
ACTUAL_ROOT = BENCHMARK_ROOT / "actual_kaggle_execution"
PIPELINE_ROOT = BENCHMARK_ROOT / "agent_pipeline"

CASE_IDS = [f"case_{index:03d}" for index in range(1, 11)]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    return value if isinstance(value, dict) else None


def direct_status(value: Any) -> str | None:
    if isinstance(value, str) and value in {"PASS", "FAIL"}:
        return value
    return None


def status_from_object(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None

    # Explicit top-level status.
    status = direct_status(value.get("status"))
    if status is not None:
        return status

    # Common verification containers.
    for key in (
        "verification",
        "candidate",
        "contract",
        "historical_execution",
        "execution",
        "compile",
        "correctness",
    ):
        nested = value.get(key)

        nested_status = direct_status(nested)
        if nested_status is not None:
            return nested_status

        if isinstance(nested, dict):
            nested_status = status_from_object(nested)
            if nested_status is not None:
                return nested_status

    # Deterministic verification counts.
    verification = value.get("verification")
    if isinstance(verification, dict):
        passed = verification.get("passed")
        failed = verification.get("failed")

        if isinstance(passed, int) and isinstance(failed, int):
            return "PASS" if failed == 0 else "FAIL"

    # Top-level passed/failed counts.
    passed = value.get("passed")
    failed = value.get("failed")

    if isinstance(passed, int) and isinstance(failed, int):
        return "PASS" if failed == 0 else "FAIL"

    return None


def first_existing(paths: list[Path]) -> tuple[Path | None, dict[str, Any] | None]:
    for path in paths:
        data = load_json(path)
        if data is not None:
            return path, data

    return None, None


def baseline_for(case_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    path = BENCHMARK_ROOT / f"{case_id}_baseline.json"
    return path, load_json(path)


def candidate_sources(case_id: str) -> list[Path]:
    actual = ACTUAL_ROOT / case_id
    pipeline = PIPELINE_ROOT / case_id

    # Case-specific evidence is deliberately explicit.
    # Historical artifacts are READ ONLY.
    sources: list[Path] = []

    if case_id == "case_001":
        sources.extend(
            [
                actual / "actual_result.json",
                actual / "case_001_cuda_boundary_result.json",
                pipeline / "case_001" / "candidate_verification" / "verifier_result.json",
                pipeline / "case_001" / "full_verification_v2" / "verifier_result.json",
            ]
        )

    elif case_id == "case_002":
        sources.extend(
            [
                actual / "existing_candidate_revalidation.json",
                actual / "iteration_2_result.json",
                actual / "iteration_3_result.json",
                pipeline / "real_interface_verification" / "verifier_result.json",
                pipeline / "final_real_contract_verification" / "verifier_result.json",
            ]
        )

    elif case_id == "case_003":
        sources.extend(
            [
                actual / "iteration_2_result.json",
                pipeline / "real_interface_verification" / "verifier_result.json",
            ]
        )

    elif case_id == "case_004":
        # Prefer the final real-contract verification produced by the
        # canonical agent pipeline. Earlier CUDA iteration artifacts are
        # historical evidence and may represent intermediate failures.
        sources.extend(
            [
                pipeline / "final_real_contract_verification" / "verifier_result.json",
                actual / "cuda_verification_iteration_3.json",
                actual / "cuda_verification.json",
                actual / "iteration_3_result.json",
            ]
        )

    elif case_id == "case_005":
        sources.extend(
            [
                pipeline / "static_lifetime_verification" / "verifier_result.json",
                pipeline / "finalization" / "finalization.json",
                actual / "static_verification.txt",
            ]
        )

    elif case_id == "case_006":
        sources.extend(
            [
                actual / "verifier_result.json",
                pipeline / "final_real_contract_verification" / "verifier_result.json",
            ]
        )

    elif case_id == "case_007":
        sources.extend(
            [
                actual / "verifier_result.json",
                pipeline / "real_contract_verification" / "verifier_result.json",
                pipeline / "finalization" / "finalization.json",
            ]
        )

    elif case_id == "case_008":
        sources.extend(
            [
                actual / "runtime_verification.json",
                pipeline / "final_real_contract_verification" / "verifier_result.json",
            ]
        )

    elif case_id == "case_009":
        sources.extend(
            [
                actual / "runtime_verification_corrected.json",
                actual / "runtime_verification.json",
                pipeline / "finalization" / "finalization.json",
            ]
        )

    elif case_id == "case_010":
        sources.extend(
            [
                actual / "runtime_verification.json",
                pipeline / "real_contract_verification" / "verifier_result.json",
                pipeline / "finalization" / "finalization.json",
            ]
        )

    return sources


def candidate_runtime(data: dict[str, Any]) -> float | None:
    for key in (
        "candidate_runtime",
        "runtime_seconds",
        "runtime",
    ):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    verification = data.get("verification")
    if isinstance(verification, dict):
        for key in (
            "candidate_runtime",
            "runtime_seconds",
            "runtime",
        ):
            value = verification.get(key)
            if isinstance(value, (int, float)):
                return float(value)

    return None


def candidate_kernel_time(data: dict[str, Any]) -> float | None:
    for key in (
        "KERNEL_TIME_MS",
        "kernel_time_ms",
        "candidate_kernel_time_ms",
    ):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    candidate = data.get("candidate")
    if isinstance(candidate, dict):
        for key in (
            "KERNEL_TIME_MS",
            "kernel_time_ms",
            "candidate_kernel_time_ms",
        ):
            value = candidate.get(key)
            if isinstance(value, (int, float)):
                return float(value)

    return None


def main() -> int:
    print("CANONICAL_EVIDENCE_RECONCILIATION=START")
    print("HISTORICAL_EVIDENCE=READ_ONLY")
    print("EXECUTION=DISABLED")
    print()

    results: list[dict[str, Any]] = []

    for case_id in CASE_IDS:
        baseline_path, baseline = baseline_for(case_id)

        baseline_status = status_from_object(baseline)

        sources = candidate_sources(case_id)

        selected_path: Path | None = None
        selected_data: dict[str, Any] | None = None
        selected_status: str | None = None

        for source in sources:
            data = load_json(source)
            if data is None:
                continue

            status = status_from_object(data)

            if status is not None:
                selected_path = source
                selected_data = data
                selected_status = status
                break

        candidate_runtime_value = (
            candidate_runtime(selected_data)
            if selected_data is not None
            else None
        )

        candidate_kernel_time_value = (
            candidate_kernel_time(selected_data)
            if selected_data is not None
            else None
        )

        baseline_runtime_value = None
        if isinstance(baseline, dict):
            for key in (
                "runtime_seconds",
                "baseline_runtime",
            ):
                value = baseline.get(key)
                if isinstance(value, (int, float)):
                    baseline_runtime_value = float(value)
                    break

        speedup = None
        if (
            baseline_runtime_value is not None
            and candidate_runtime_value is not None
            and candidate_runtime_value > 0
        ):
            speedup = baseline_runtime_value / candidate_runtime_value

        result = {
            "case_id": case_id,
            "baseline_correctness": baseline_status,
            "candidate_correctness": selected_status,
            "baseline_runtime": baseline_runtime_value,
            "candidate_runtime": candidate_runtime_value,
            "candidate_kernel_time_ms": candidate_kernel_time_value,
            "speedup": speedup,
            "candidate_evidence": (
                str(selected_path.relative_to(ROOT))
                if selected_path is not None
                else None
            ),
            "baseline_evidence": (
                str(baseline_path.relative_to(ROOT))
                if baseline_path is not None
                else None
            ),
        }

        results.append(result)

        print(
            f"{case_id} | "
            f"baseline={baseline_status} | "
            f"candidate={selected_status} | "
            f"candidate_evidence="
            f"{result['candidate_evidence']}"
        )

    print()
    print("===== AGGREGATE =====")

    baseline_passed = sum(
        item["baseline_correctness"] == "PASS"
        for item in results
    )

    candidate_passed = sum(
        item["candidate_correctness"] == "PASS"
        for item in results
    )

    candidate_missing = [
        item["case_id"]
        for item in results
        if item["candidate_correctness"] is None
    ]

    candidate_failed = [
        item["case_id"]
        for item in results
        if item["candidate_correctness"] == "FAIL"
    ]

    print(f"cases={len(results)}")
    print(f"baseline_passed={baseline_passed}")
    print(f"candidate_passed={candidate_passed}")
    print(
        "candidate_success_rate="
        f"{candidate_passed / len(results):.4f}"
        if results
        else "candidate_success_rate=0.0000"
    )
    print(f"candidate_missing={candidate_missing}")
    print(f"candidate_failed={candidate_failed}")

    print()
    print("===== SAFETY =====")
    print("REPORT_WRITTEN=NO")
    print("TRAJECTORIES_WRITTEN=NO")
    print("BASELINES_WRITTEN=NO")
    print("GEMINI_EXECUTION=NOT_PERFORMED")
    print("CUDA_EXECUTION=NOT_PERFORMED")
    print("KAGGLE_EXECUTION=NOT_PERFORMED")
    print("GIT_OPERATION=NONE")

    unresolved = candidate_missing

    if unresolved:
        print()
        print("RECONCILIATION_STATUS=INCOMPLETE")
        print(
            "UNRESOLVED_CASES="
            + ",".join(unresolved)
        )
        return 2

    if candidate_failed:
        print()
        print("RECONCILIATION_STATUS=EVIDENCE_FAILURE")
        print(
            "FAILED_CASES="
            + ",".join(candidate_failed)
        )
        return 3

    if baseline_passed != len(results):
        print()
        print("RECONCILIATION_STATUS=BASELINE_FAILURE")
        return 4

    print()
    print("RECONCILIATION_STATUS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
