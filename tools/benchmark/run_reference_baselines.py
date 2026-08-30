from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
CASES_ROOT = ROOT / "cases"
BENCHMARK_ROOT = ROOT / "trajectories" / "benchmark"
EXECUTION_ROOT = ROOT / "trajectories" / "execution"

BASELINE_CASES = [
    "case_002",
    "case_003",
    "case_004",
    "case_005",
    "case_006",
    "case_007",
    "case_008",
    "case_009",
    "case_010",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_metadata(case_id: str) -> dict[str, Any]:
    return load_json(CASES_ROOT / case_id / "metadata.json")


def read_case_sources(case_id: str) -> str:
    case_root = CASES_ROOT / case_id
    parts: list[str] = []

    for directory in (
        case_root / "reference",
        case_root / "tests",
    ):
        if not directory.is_dir():
            continue

        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".cu", ".md"}:
                parts.append(
                    path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                )

    return "\n".join(parts)


def classify(metadata: dict[str, Any], source: str) -> str:
    runtime = metadata.get("required_runtime", [])
    if not isinstance(runtime, list):
        runtime = []

    text = " ".join(
        [
            str(metadata.get("domain", "")),
            str(metadata.get("name", "")),
            str(metadata.get("verification_strategy", "")),
            " ".join(str(x) for x in runtime),
            source,
        ]
    ).lower()

    if "distributed" in text:
        return "DISTRIBUTED_STATIC"

    if "async" in text and "cuda" not in text:
        return "ASYNC_PYTHON"

    if "pytorch" in text or "torch" in text:
        return "PYTORCH"

    if "cuda" in text:
        return "CUDA"

    return "STATIC"


def run_command(
    command: list[str],
    cwd: Path,
    timeout: int = 120,
) -> tuple[int, str, str, float]:
    started = time.perf_counter()

    process = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    elapsed = time.perf_counter() - started

    return (
        process.returncode,
        process.stdout,
        process.stderr,
        elapsed,
    )


def static_contract(case_id: str) -> dict[str, Any]:
    case_root = CASES_ROOT / case_id
    tests_dir = case_root / "tests"

    commands: list[list[str]] = []

    if (tests_dir / "test_contract.py").is_file():
        commands.append(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(tests_dir / "test_contract.py"),
            ]
        )

    if (tests_dir / "test_harness_contract.py").is_file():
        commands.append(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(tests_dir / "test_harness_contract.py"),
            ]
        )

    outputs = []
    total_runtime = 0.0

    for command in commands:
        code, stdout, stderr, elapsed = run_command(
            command,
            ROOT,
        )

        total_runtime += elapsed

        outputs.append(
            {
                "command": command,
                "return_code": code,
                "stdout": stdout[-4000:],
                "stderr": stderr[-4000:],
                "runtime_seconds": round(elapsed, 6),
            }
        )

        if code != 0:
            return {
                "status": "FAIL",
                "correctness": "FAIL",
                "runtime_seconds": round(total_runtime, 6),
                "execution": {
                    "cuda": False,
                    "gemini": False,
                    "mode": "static_contract",
                },
                "checks": outputs,
            }

    return {
        "status": "PASS",
        "correctness": "PASS",
        "runtime_seconds": round(total_runtime, 6),
        "execution": {
            "cuda": False,
            "gemini": False,
            "mode": "static_contract",
        },
        "checks": outputs,
    }


def python_reference(case_id: str) -> dict[str, Any]:
    case_root = CASES_ROOT / case_id
    reference_files = sorted(
        case_root.joinpath("reference").glob("*.py")
    )

    if not reference_files:
        return static_contract(case_id)

    reference = reference_files[0]

    code = (
        "import runpy\n"
        f"runpy.run_path({str(reference)!r}, run_name='__main__')\n"
    )

    return_code, stdout, stderr, elapsed = run_command(
        [sys.executable, "-c", code],
        ROOT,
        timeout=120,
    )

    if return_code == 0:
        status = "PASS"
        correctness = "PASS"
    else:
        status = "FAIL"
        correctness = "FAIL"

    return {
        "status": status,
        "correctness": correctness,
        "runtime_seconds": round(elapsed, 6),
        "execution": {
            "cuda": False,
            "gemini": False,
            "mode": "python_reference",
        },
        "reference": str(reference.relative_to(ROOT)),
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
    }


def cuda_compile_only(case_id: str) -> dict[str, Any]:
    case_root = CASES_ROOT / case_id
    reference_dir = case_root / "reference"

    sources = sorted(reference_dir.glob("*.cu"))

    if not sources:
        return static_contract(case_id)

    source = sources[0]
    run_dir = ROOT / "trajectories" / "benchmark" / "reference_runs" / case_id
    run_dir.mkdir(parents=True, exist_ok=True)

    copied_source = run_dir / source.name
    copied_source.write_text(
        source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    executable = run_dir / "reference_compile_check"

    code, stdout, stderr, elapsed = run_command(
        [
            "nvcc",
            "-std=c++17",
            "-arch=sm_75",
            "-c",
            str(copied_source),
            "-o",
            str(executable.with_suffix(".o")),
        ],
        ROOT,
        timeout=120,
    )

    return {
        "status": "PASS" if code == 0 else "FAIL",
        "correctness": "PASS" if code == 0 else "FAIL",
        "runtime_seconds": round(elapsed, 6),
        "execution": {
            "cuda": True,
            "gemini": False,
            "mode": "cuda_reference_compile",
        },
        "reference": str(source.relative_to(ROOT)),
        "compile": {
            "return_code": code,
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
        },
    }


def execute_case(case_id: str) -> dict[str, Any]:
    metadata = load_metadata(case_id)
    source = read_case_sources(case_id)
    mode = classify(metadata, source)

    if mode == "DISTRIBUTED_STATIC":
        result = static_contract(case_id)
    elif mode == "ASYNC_PYTHON":
        result = static_contract(case_id)
    elif mode == "PYTORCH":
        result = static_contract(case_id)
    elif mode == "CUDA":
        result = cuda_compile_only(case_id)
    else:
        result = static_contract(case_id)

    result["case_id"] = case_id
    result["implementation"] = "reference"
    result["execution_class"] = mode
    result["gemini_api_call"] = False

    return result


def main() -> int:
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    EXECUTION_ROOT.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("GPU ENGINEERING AGENT BENCHMARK")
    print("KAGGLE — REFERENCE BASELINE BATCH 002-010")
    print("============================================================")
    print()
    print("POLICY=CASE_SPECIFIC")
    print("CASE_001=PROTECTED")
    print("GEMINI_API_CALL=NONE")
    print()

    results: list[dict[str, Any]] = []

    for case_id in BASELINE_CASES:
        print(f"===== {case_id} =====")

        try:
            result = execute_case(case_id)
        except Exception as exc:
            metadata = load_metadata(case_id)
            source = read_case_sources(case_id)
            execution_class = classify(metadata, source)

            result = {
                "case_id": case_id,
                "implementation": "reference",
                "status": "FAIL",
                "correctness": "FAIL",
                "runtime_seconds": None,
                "execution": {
                    "cuda": "cuda" in metadata.get("required_runtime", []),
                    "gemini": False,
                    "mode": "executor_error",
                },
                "execution_class": execution_class,
                "error": f"{type(exc).__name__}: {exc}",
                "gemini_api_call": False,
            }

        results.append(result)

        baseline_path = (
            BENCHMARK_ROOT
            / f"{case_id}_baseline.json"
        )

        trajectory_path = (
            EXECUTION_ROOT
            / f"{case_id}.json"
        )

        write_json(
            baseline_path,
            {
                "schema_version": "1.0",
                **result,
            },
        )

        write_json(
            trajectory_path,
            {
                "schema_version": "1.0",
                "case_id": case_id,
                "mode": "reference_baseline",
                "baseline": {
                    "implementation": "reference",
                    "status": result["status"],
                    "correctness": result["correctness"],
                    "runtime_seconds": result.get(
                        "runtime_seconds"
                    ),
                },
                "candidate": {
                    "implementation": "gemini_flash_agent",
                    "status": "PENDING",
                },
                "verifier": {
                    "status": "PENDING",
                },
                "trajectory": {
                    "status": (
                        "READY"
                        if result["status"] == "PASS"
                        else "FAILED"
                    ),
                    "execution_class": result[
                        "execution_class"
                    ],
                },
                "secrets": {
                    "api_keys_recorded": False,
                    "credentials_recorded": False,
                },
            },
        )

        print(
            f"{case_id}: {result['status']}"
        )
        print(
            f"MODE={result['execution_class']}"
        )
        print(
            f"BASELINE={baseline_path}"
        )
        print(
            f"TRAJECTORY={trajectory_path}"
        )
        print()

    passed = sum(
        result["status"] == "PASS"
        for result in results
    )

    failed = len(results) - passed

    report = {
        "schema_version": "1.0",
        "benchmark": "gpu-engineering-agent-benchmark",
        "mode": "reference_baseline_batch",
        "cases": len(results),
        "passed": passed,
        "failed": failed,
        "results": results,
        "execution": {
            "cuda": any(
                result["execution"].get("cuda")
                for result in results
            ),
            "gemini": False,
            "batch": True,
        },
    }

    report_path = (
        BENCHMARK_ROOT
        / "reference_baseline_results.json"
    )

    write_json(report_path, report)

    print("============================================================")
    print("REFERENCE BASELINE BATCH COMPLETE")
    print("============================================================")
    print(f"CASES={len(results)}")
    print(f"PASSED={passed}")
    print(f"FAILED={failed}")
    print(f"REPORT={report_path}")
    print("CASE_001=PROTECTED")
    print("GEMINI_API_CALL=NONE")
    print("TERMINAL_CLOSE=NONE")
    print("============================================================")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
