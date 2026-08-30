from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "tools" / "batch" / "case_registry.json"
OUTPUT = ROOT / "trajectories" / "batch" / "static_results.json"

registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

results: list[dict[str, object]] = []

for case in registry["cases"]:
    case_id = case["case_id"]
    case_path = ROOT / case["path"]
    test_path = case_path / "tests"

    started = time.monotonic()

    completed = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            str(test_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={
            **__import__("os").environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT),
        },
    )

    duration = time.monotonic() - started

    output = completed.stdout + completed.stderr

    status = "PASS" if completed.returncode == 0 else "FAIL"

    results.append(
        {
            "case_id": case_id,
            "status": status,
            "return_code": completed.returncode,
            "duration_seconds": round(duration, 4),
            "tests_passed": output.count(" passed"),
            "tests_failed": output.count(" failed"),
            "tests_error": output.count(" error"),
            "trajectory": str(case_path / "trajectory"),
        }
    )

    print(f"{case_id}: {status}")

OUTPUT.write_text(
    json.dumps(
        {
            "mode": "static-batch",
            "cuda_execution": False,
            "gemini_api": False,
            "kaggle": False,
            "results": results,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

passed = sum(result["status"] == "PASS" for result in results)

print()
print("============================================================")
print("STATIC BATCH COMPLETE")
print("============================================================")
print(f"CASES={len(results)}")
print(f"PASSED={passed}")
print(f"FAILED={len(results) - passed}")
print(f"REPORT={OUTPUT}")
