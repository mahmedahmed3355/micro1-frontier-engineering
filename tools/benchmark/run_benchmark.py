from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = ROOT / "cases"
OUTPUT_ROOT = ROOT / "trajectories" / "execution"
BENCHMARK_ROOT = ROOT / "trajectories" / "benchmark"
REGISTRY_PATH = ROOT / "tools" / "batch" / "case_registry.json"


def load_registry() -> list[dict[str, Any]]:
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        cases = data.get("cases")
    else:
        cases = data

    if not isinstance(cases, list):
        raise ValueError("case registry must contain a list of cases")

    return cases


def case_id_from_entry(entry: dict[str, Any]) -> str:
    for key in ("case_id", "id", "name"):
        value = entry.get(key)
        if isinstance(value, str):
            return value

    raise ValueError("registry entry has no case identifier")


def verify_case_layout(case_id: str) -> None:
    case_root = CASES_ROOT / case_id

    required = (
        "metadata.json",
        "README.md",
        "src",
        "reference",
        "tests",
        "trajectory",
    )

    for relative_path in required:
        path = case_root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"{case_id}: missing {relative_path}")


def run_static_dry_run(case_id: str) -> dict[str, Any]:
    started = time.perf_counter()

    verify_case_layout(case_id)

    case_root = CASES_ROOT / case_id
    metadata_path = case_root / "metadata.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    elapsed = time.perf_counter() - started

    return {
        "case_id": case_id,
        "mode": "dry_run",
        "baseline": {"status": "PENDING_CUDA"},
        "candidate": {"status": "PENDING_GEMINI"},
        "verifier": {"status": "PENDING_CUDA"},
        "trajectory": {"status": "READY"},
        "metadata_loaded": bool(metadata),
        "elapsed_seconds": round(elapsed, 6),
    }


def build_report(results: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "benchmark": "gpu-engineering-agent-benchmark",
        "mode": mode,
        "cases": len(results),
        "results": results,
        "execution": {
            "cuda": mode == "cuda",
            "gemini": mode == "cuda",
            "batch": True,
        },
    }


def write_case_trajectory(result: dict[str, Any]) -> None:
    case_id = result["case_id"]
    path = OUTPUT_ROOT / f"{case_id}.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("dry-run",),
        default="dry-run",
    )
    args = parser.parse_args()

    registry = load_registry()
    results: list[dict[str, Any]] = []

    for entry in registry:
        case_id = case_id_from_entry(entry)
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


if __name__ == "__main__":
    raise SystemExit(main())
