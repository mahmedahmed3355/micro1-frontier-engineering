from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = ROOT / "cases"
BENCHMARK_ROOT = ROOT / "trajectories" / "benchmark"
REGISTRY_PATH = ROOT / "tools" / "batch" / "case_registry.json"

CASE_COUNT = 10


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return value


def load_registry() -> list[dict[str, Any]]:
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        cases = data.get("cases")
    else:
        cases = data

    if not isinstance(cases, list):
        raise ValueError("case registry must contain a list")

    return cases


def case_id_from_entry(entry: dict[str, Any]) -> str:
    value = entry.get("case_id")

    if not isinstance(value, str):
        raise ValueError("registry entry has no case_id")

    return value


def load_metadata(case_id: str) -> dict[str, Any]:
    return load_json(CASES_ROOT / case_id / "metadata.json")


def baseline_path(case_id: str) -> Path:
    return BENCHMARK_ROOT / f"{case_id}_baseline.json"


def historical_trajectory_path(case_id: str) -> Path:
    return ROOT / "trajectories" / "execution" / f"{case_id}.json"


def actual_execution_dir(case_id: str) -> Path:
    return BENCHMARK_ROOT / "actual_kaggle_execution" / case_id


def discover_actual_artifacts(case_id: str) -> list[str]:
    directory = actual_execution_dir(case_id)

    if not directory.is_dir():
        return []

    return sorted(
        str(path.relative_to(ROOT))
        for path in directory.rglob("*")
        if path.is_file()
    )


def classify_case(metadata: dict[str, Any]) -> str:
    required = set(metadata.get("required_runtime", []))

    if "cuda" in required and "torch" not in required:
        return "CUDA"

    if "distributed" in required:
        return "DISTRIBUTED"

    if "torch" in required:
        return "PYTORCH"

    if "python" in required:
        return "PYTHON"

    return "UNKNOWN"


def inspect_case(entry: dict[str, Any]) -> dict[str, Any]:
    case_id = case_id_from_entry(entry)
    metadata = load_metadata(case_id)

    baseline_file = baseline_path(case_id)
    trajectory_file = historical_trajectory_path(case_id)

    baseline: dict[str, Any] | None = None

    if baseline_file.is_file():
        baseline = load_json(baseline_file)

    required_runtime = set(metadata.get("required_runtime", []))

    cuda_policy = metadata.get("cuda_execution_required")
    if cuda_policy is None:
        cuda_required = "cuda" in required_runtime
    else:
        cuda_required = bool(cuda_policy)

    gemini_required = metadata.get("gemini_api_required")

    return {
        "case_id": case_id,
        "name": metadata.get("name"),
        "domain": metadata.get("domain"),
        "bug_class": metadata.get("bug_class"),
        "execution_class": classify_case(metadata),
        "required_runtime": metadata.get("required_runtime", []),
        "gemini_required": gemini_required,
        "cuda_required": cuda_required,
        "baseline": {
            "path": str(baseline_file.relative_to(ROOT)),
            "found": baseline is not None,
            "status": (
                baseline.get("status")
                if baseline is not None
                else "MISSING"
            ),
            "correctness": (
                baseline.get("correctness")
                if baseline is not None
                else None
            ),
            "runtime_seconds": (
                baseline.get("runtime_seconds")
                if baseline is not None
                else None
            ),
        },
        "historical_trajectory": {
            "path": str(trajectory_file.relative_to(ROOT)),
            "found": trajectory_file.is_file(),
        },
        "actual_execution_artifacts": discover_actual_artifacts(case_id),
    }


def build_report(
    results: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    baseline_passed = sum(
        result["baseline"]["correctness"] == "PASS"
        for result in results
    )

    baseline_failed = sum(
        result["baseline"]["correctness"] == "FAIL"
        for result in results
    )

    return {
        "schema_version": "1.0",
        "benchmark": "gpu-engineering-agent-benchmark",
        "mode": mode,
        "cases": len(results),
        "baseline": {
            "implementation": "reference",
            "passed": baseline_passed,
            "failed": baseline_failed,
            "complete": len(results) == CASE_COUNT,
        },
        "candidate": {
            "implementation": "gemini_flash_agent",
            "status": "PENDING_FINAL_EXECUTION",
        },
        "verifier": {
            "status": "PENDING_FINAL_EXECUTION",
        },
        "trajectory": {
            "historical_artifacts_preserved": True,
            "status": "PRESERVED",
        },
        "results": results,
        "execution": {
            "batch": True,
            "cuda": any(
                result["cuda_required"] is True
                for result in results
            ),
            "gemini": any(
                result["gemini_required"] is True
                for result in results
            ),
        },
        "aggregate": {
            "cases": len(results),
            "baseline_passed": baseline_passed,
            "baseline_failed": baseline_failed,
            "candidate_passed": None,
            "candidate_success_rate": None,
            "average_baseline_runtime": None,
            "average_candidate_runtime": None,
            "average_speedup": None,
            "failed_cases": [],
        },
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_agent_pipeline(case_id: str) -> dict[str, Any]:
    """
    Execute the existing project agent pipeline for one case.

    This function adapts the canonical AgentState contract to the
    existing Reader -> Analyzer -> Optimizer -> Verifier pipeline.
    It does not implement a second agent pipeline.
    """
    from agents.reader.gemini_reader import GeminiReaderAgent
    from agents.analyzer.agent import AnalyzerAgent
    from agents.optimizer.agent import OptimizerAgent
    from agents.verifier.agent import VerifierAgent

    case_root = CASES_ROOT / case_id
    metadata = json.loads(
        (case_root / "metadata.json").read_text(encoding="utf-8")
    )

    input_files: list[str] = []

    for subdir in ("src", "reference", "tests"):
        base = case_root / subdir
        if base.exists():
            for item in sorted(base.rglob("*")):
                if item.is_file():
                    input_files.append(
                        str(item.relative_to(case_root))
                    )

    source_candidates = [
        item
        for item in sorted((case_root / "src").glob("*"))
        if item.is_file()
    ]

    if not source_candidates:
        raise FileNotFoundError(
            f"{case_id}: no source file found under {case_root / 'src'}"
        )

    input_source = source_candidates[0].read_text(
        encoding="utf-8"
    )

    baseline_file = baseline_path(case_id)
    baseline_evidence: dict[str, Any] = {}

    if baseline_file.is_file():
        baseline_evidence = load_json(baseline_file)

    state: dict[str, Any] = {
        "case_id": case_id,
        "case_dir": str(case_root),
        "workspace": str(case_root),
        "input_files": input_files,
        "input_source": input_source,
        "baseline_evidence": baseline_evidence,
        "metadata": metadata,
        "iteration": 0,
        "max_iterations": 1,
        "status": "running",
        "feedback": [],
        "feedback_history": [],
        "optimization_history": [],
        "rejected_candidate_hashes": [],
    }

    result: dict[str, Any] = {
        "case_id": case_id,
        "mode": "agent_pipeline",
        "status": "ERROR",
        "pipeline": [],
    }

    try:
        llm = build_llm_client()

        reader = GeminiReaderAgent(llm=llm)
        reader_result = reader.run(state)
        state.update(reader_result)

        result["pipeline"].append({
            "agent": "reader",
            "status": "PASS",
        })

        analyzer = AnalyzerAgent(llm=llm)
        analyzer_result = analyzer.run(state)
        state.update(analyzer_result)

        result["pipeline"].append({
            "agent": "analyzer",
            "status": "PASS",
        })

        optimizer = OptimizerAgent(llm=llm)
        optimizer_result = optimizer.run(state)
        state.update(optimizer_result)

        result["pipeline"].append({
            "agent": "optimizer",
            "status": "PASS",
        })

        verifier = VerifierAgent()
        verifier_result = verifier.run(state)
        state.update(verifier_result)

        result["pipeline"].append({
            "agent": "verifier",
            "status": "PASS",
        })

        verification = state.get("verification", {})

        result.update({
            "status": (
                "PASS"
                if state.get("accepted")
                else "FAIL"
            ),
            "baseline": {
                "implementation": "reference",
                "status": (
                    "PASS"
                    if baseline_evidence.get("status") == "PASS"
                    else baseline_evidence.get("status")
                ),
                "correctness": baseline_evidence.get("correctness"),
                "runtime_seconds": baseline_evidence.get(
                    "runtime_seconds"
                ),
            },
            "candidate": {
                "implementation": "gemini_flash_agent",
                "status": (
                    "PASS"
                    if state.get("accepted")
                    else "FAIL"
                ),
                "correctness": (
                    verification.get("correctness", {}).get("passed")
                    if isinstance(verification.get("correctness"), dict)
                    else None
                ),
                "runtime_seconds": (
                    verification.get("performance", {}).get("candidate_ms")
                    if isinstance(verification.get("performance"), dict)
                    else None
                ),
                "kernel_time_ms": (
                    verification.get("performance", {}).get("candidate_ms")
                    if isinstance(verification.get("performance"), dict)
                    else None
                ),
            },
            "verifier": verification,
            "trajectory": {
                "status": "RECORDED",
            },
        })

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def build_llm_client():
    """
    Construct the existing Gemini adapter.

    The benchmark runner does not implement a new LLM client.
    """
    from llm.gemini import GeminiFlashClient

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not available")

    model = os.environ.get(
        "GEMINI_MODEL",
        "gemini-3.6-flash",
    )

    return GeminiFlashClient(
        api_key=api_key,
        model=model,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("dry-run", "agent"),
        default="dry-run",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only the specified case ID. May be repeated.",
    )
    args = parser.parse_args()

    registry = load_registry()

    selected_cases = {
        case_id_from_entry(entry)
        for entry in registry
    }

    if args.cases:
        requested = set(args.cases)
        unknown = requested - selected_cases
        if unknown:
            raise SystemExit(
                f"Unknown case(s): {', '.join(sorted(unknown))}"
            )
        selected_cases &= requested

    ordered_cases = [
        case_id_from_entry(entry)
        for entry in registry
        if case_id_from_entry(entry) in selected_cases
    ]

    if args.mode == "dry-run":
        results: list[dict[str, Any]] = []

        for case_id in ordered_cases:
            result = run_static_dry_run(case_id)
            results.append(result)
            write_case_trajectory(result)
            print(f"{case_id}: READY")

        report = build_report(results, args.mode)

        report_path = BENCHMARK_ROOT / "dry_run_results.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print(f"CASES={len(results)}")
        print(f"REPORT={report_path}")
        print("DRY_RUN=PASS")

        return 0

    results = []

    for case_id in ordered_cases:
        print()
        print("=" * 60)
        print(case_id)
        print("=" * 60)

        result = run_agent_pipeline(case_id)
        results.append(result)

        print(
            f"{case_id}: "
            f"{result.get('status', 'ERROR')}"
        )

    report = {
        "schema_version": "1.0",
        "benchmark": "gpu-engineering-agent-benchmark",
        "mode": "agent_pipeline",
        "cases": len(results),
        "results": results,
        "execution": {
            "cuda": True,
            "gemini": True,
            "batch": True,
        },
    }

    report_path = BENCHMARK_ROOT / "agent_pipeline_results.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"CASES={len(results)}")
    print(f"REPORT={report_path}")

    return 0 if all(
        result.get("status") == "PASS"
        for result in results
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
