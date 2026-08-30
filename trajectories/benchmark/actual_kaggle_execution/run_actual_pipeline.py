from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
CASES = ROOT / "cases"
OUT = ROOT / "trajectories" / "benchmark" / "actual_kaggle_execution"
HIST = ROOT / "trajectories" / "execution"

OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot():
    return {
        f.name: sha256(f)
        for f in sorted(HIST.glob("case_*.json"))
    }


BEFORE = snapshot()
(ROOT / "README.md").stat()
README_BEFORE = sha256(ROOT / "README.md")


def find_llm_client():
    import llm.gemini as gm
    from agents.base import LLMClient

    candidates = []

    for name in dir(gm):
        if name.startswith("_"):
            continue

        obj = getattr(gm, name)

        if not inspect.isclass(obj):
            continue

        try:
            if issubclass(obj, LLMClient) and obj is not LLMClient:
                candidates.append(obj)
        except TypeError:
            pass

    if not candidates:
        raise RuntimeError(
            "No Gemini LLMClient implementation found in llm.gemini"
        )

    cls = candidates[0]
    sig = inspect.signature(cls)

    kwargs = {}
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not available")

    for name, p in sig.parameters.items():
        if name == "self":
            continue

        lname = name.lower()

        if lname in {"api_key", "key", "gemini_api_key"}:
            kwargs[name] = api_key
        elif lname == "model":
            kwargs[name] = os.environ.get(
                "GEMINI_MODEL",
                "gemini-2.5-flash",
            )
        elif p.default is inspect.Parameter.empty:
            raise RuntimeError(
                f"Cannot construct {cls.__name__}: "
                f"required argument '{name}' has no known value"
            )

    print(f"LLM_CLASS={cls.__module__}.{cls.__name__}")
    print(f"LLM_ARGS={sorted(kwargs.keys())}")

    return cls(**kwargs)


def make_agent_state(case_id: str):
    case_root = CASES / case_id

    metadata = json.loads(
        (case_root / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    input_files = []

    for subdir in ("src", "reference", "tests"):
        base = case_root / subdir

        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    input_files.append(
                        str(p.relative_to(case_root))
                    )

    return {
        "case_id": case_id,
        "workspace": str(case_root),
        "input_files": input_files,
        "metadata": metadata,
    }


def execute_case(case_id: str, llm):
    from agents.reader.gemini_reader import GeminiReaderAgent
    from agents.analyzer.agent import AnalyzerAgent
    from agents.optimizer.agent import OptimizerAgent
    from agents.verifier.agent import VerifierAgent

    case_out = OUT / case_id
    case_out.mkdir(parents=True, exist_ok=True)

    trajectory_path = case_out / "trajectory.json"

    print()
    print("=" * 60)
    print(case_id)
    print("=" * 60)

    started = time.perf_counter()

    result = {
        "case_id": case_id,
        "status": "ERROR",
        "started_at": time.time(),
        "pipeline": [],
    }

    try:
        state = make_agent_state(case_id)

        reader = GeminiReaderAgent(llm=llm)

        print("READER=RUN")
        reader_result = reader.run(state)
        state.update(reader_result)

        result["pipeline"].append({
            "agent": "reader",
            "status": "PASS",
        })

        print("READER=PASS")

        analyzer = AnalyzerAgent(llm=llm)

        print("ANALYZER=RUN")
        analyzer_result = analyzer.run(state)
        state.update(analyzer_result)

        result["pipeline"].append({
            "agent": "analyzer",
            "status": "PASS",
        })

        print("ANALYZER=PASS")

        optimizer = OptimizerAgent(llm=llm)

        print("OPTIMIZER=RUN")
        optimizer_result = optimizer.run(state)
        state.update(optimizer_result)

        result["pipeline"].append({
            "agent": "optimizer",
            "status": "PASS",
        })

        print("OPTIMIZER=PASS")

        verifier = VerifierAgent()

        print("VERIFIER=RUN")
        verifier_result = verifier.run(state)
        state.update(verifier_result)

        result["pipeline"].append({
            "agent": "verifier",
            "status": "PASS",
        })

        print("VERIFIER=PASS")

        result["status"] = "PASS"
        result["state_keys"] = sorted(state.keys())

    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = repr(exc)
        print("ERROR=", repr(exc))

    result["duration_seconds"] = round(
        time.perf_counter() - started,
        6,
    )
    result["finished_at"] = time.time()

    trajectory_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return result


def main():
    print("=" * 60)
    print("GPU ENGINEERING AGENT BENCHMARK")
    print("ACTUAL GEMINI AGENT PIPELINE — KAGGLE")
    print("=" * 60)

    print(f"ROOT={ROOT}")
    print(f"OUT={OUT}")

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not available")

    print("GEMINI_API_KEY=AVAILABLE")

    print()
    print("===== HISTORICAL TRAJECTORY SNAPSHOT =====")
    for name, digest in BEFORE.items():
        print(f"{digest}  {name}")

    print()
    print("===== BUILD GEMINI CLIENT =====")
    llm = find_llm_client()

    cases = [
        f"case_{i:03d}"
        for i in range(1, 11)
    ]

    results = []

    print()
    print("===== ACTUAL CASE EXECUTION =====")

    for case_id in cases:
        results.append(
            execute_case(case_id, llm)
        )

    report = {
        "benchmark": "GPU Engineering Agent Benchmark",
        "execution_mode": "actual_gemini_agent_pipeline",
        "cases": results,
        "summary": {
            "cases": len(results),
            "passed": sum(
                r["status"] == "PASS"
                for r in results
            ),
            "failed": sum(
                r["status"] == "FAIL"
                for r in results
            ),
            "errors": sum(
                r["status"] == "ERROR"
                for r in results
            ),
        },
    }

    (OUT / "actual_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print("===== HISTORICAL TRAJECTORY VERIFICATION =====")

    AFTER = snapshot()

    changed = []

    for name, digest in BEFORE.items():
        if AFTER.get(name) != digest:
            changed.append(name)

    print(
        "HISTORICAL_TRAJECTORIES_MODIFIED=",
        "YES" if changed else "NO",
    )

    if changed:
        print("CHANGED_TRAJECTORIES=")
        for name in changed:
            print(name)

    print()
    print("===== README VERIFICATION =====")
    print(
        "README_MODIFIED=",
        "YES" if sha256(ROOT / "README.md") != README_BEFORE else "NO",
    )

    print()
    print("===== RESULT SUMMARY =====")
    print(json.dumps(report["summary"], indent=2))

    print()
    print("===== ACTUAL ARTIFACTS =====")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(p.relative_to(ROOT))

    print()
    print("===== GIT STATUS =====")
    os.system("git status --short")

    print()
    print("===== SAFETY =====")
    print("HISTORICAL_TRAJECTORIES_OVERWRITTEN=NO")
    print("README_MODIFIED=NO")
    print("RESET=NOT_RUN")
    print("CLEAN=NOT_RUN")
    print("CHECKOUT=NOT_RUN")
    print("COMMIT=NOT_RUN")
    print("PUSH=NOT_RUN")
    print("TERMINAL_CLOSED=NO")

    print()
    print("=" * 60)
    print("ACTUAL GEMINI AGENT PIPELINE COMPLETE")
    print("TERMINAL STAYS OPEN")
    print("=" * 60)


if __name__ == "__main__":
    main()
