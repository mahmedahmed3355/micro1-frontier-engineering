from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_status = baseline.get("status")
    candidate_status = candidate.get("status")

    baseline_runtime = baseline.get("runtime_seconds")
    candidate_runtime = candidate.get("runtime_seconds")

    speedup = None

    if (
        isinstance(baseline_runtime, (int, float))
        and isinstance(candidate_runtime, (int, float))
        and candidate_runtime > 0
    ):
        speedup = baseline_runtime / candidate_runtime

    return {
        "baseline_status": baseline_status,
        "candidate_status": candidate_status,
        "correctness_match": (baseline_status == "PASS" and candidate_status == "PASS"),
        "baseline_runtime_seconds": baseline_runtime,
        "candidate_runtime_seconds": candidate_runtime,
        "speedup": speedup,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.baseline or not args.candidate:
        print("COMPARATOR=READY")
        print("MODE=CONTRACT_ONLY")
        return 0

    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)

    result = compare(baseline, candidate)

    if args.output:
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(result, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
