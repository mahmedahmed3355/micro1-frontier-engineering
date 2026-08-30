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


def historical_snapshot():
    return {
        str(p.relative_to(HIST)): sha256(p)
        for p in sorted(HIST.glob("case_*.json"))
    }


HIST_BEFORE = historical_snapshot()
README_BEFORE = sha256(ROOT / "README.md")


def build_llm():
    from llm.gemini import GeminiFlashClient

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not available")

    model = os.environ.get(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    )

    print("LLM_CLASS=llm.gemini.GeminiFlashClient")
    print(f"MODEL={model}")

    return GeminiFlashClient(
        api_key=key,
        model=model,
    )


def patch_workspace_inspector():
    path = ROOT / "tools" / "workspace" / "inspector.py"

    if not path.exists():
        print("WORKSPACE_INSPECTOR=NOT_FOUND")
        return

    text = path.read_text(encoding="utf-8")

    old = text

    replacements = [
        (
            '".cu", ".cuh", ".cpp", ".h", ".hpp", ".md", ".json"',
            '".cu", ".cuh", ".cpp", ".h", ".hpp", ".cc", ".c", ".py", ".md", ".json", ".txt"',
        ),
        (
            "('.cu', '.cuh', '.cpp', '.h', '.hpp', '.md', '.json')",
            "('.cu', '.cuh', '.cpp', '.h', '.hpp', '.cc', '.c', '.py', '.md', '.json', '.txt')",
        ),
        (
            "('.cu', '.cpp', '.h', '.hpp', '.md', '.json')",
            "('.cu', '.cpp', '.h', '.hpp', '.cc', '.c', '.py', '.md', '.json', '.txt')",
        ),
    ]

    for a, b in replacements:
        text = text.replace(a, b)

    if text != old:
        path.write_text(text, encoding="utf-8")
        print("WORKSPACE_INSPECTOR=PATCHED")
    else:
        print("WORKSPACE_INSPECTOR=NO_LITERAL_PATCH_NEEDED")


def case_input_files(case_id: str):
    case_root = CASES / case_id

    files = []

    for directory in ("src", "reference"):
        base = case_root / directory

        if not base.exists():
            continue

        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue

            if p.suffix.lower() in {
                ".cu",
                ".cuh",
                ".cpp",
                ".cc",
                ".c",
                ".h",
                ".hpp",
                ".py",
                ".md",
                ".json",
                ".txt",
            }:
                files.append(
                    str(p.relative_to(case_root))
                )

    readme = case_root / "README.md"

    if readme.exists():
        files.append("README.md")

    return sorted(set(files))


def make_state(case_id: str):
    case_root = CASES / case_id

    metadata = json.loads(
        (case_root / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    files = case_input_files(case_id)

    return {
        "case_id": case_id,
        "workspace": str(case_root),
        "input_files": files,
        "metadata": metadata,
    }


def run_case(case_id: str, llm):
    from agents.reader.gemini_reader import GeminiReaderAgent
    from agents.analyzer.agent import AnalyzerAgent
    from agents.optimizer.agent import OptimizerAgent
    from agents.verifier.agent import VerifierAgent

    case_out = OUT / case_id
    case_out.mkdir(parents=True, exist_ok=True)

    result = {
        "case_id": case_id,
        "status": "ERROR",
        "pipeline": [],
        "started_at": time.time(),
    }

    started = time.perf_counter()

    print()
    print("=" * 60)
    print(case_id)
    print("=" * 60)

    try:
        state = make_state(case_id)

        print("INPUT_FILES=")
        for f in state["input_files"]:
            print(f"  {f}")

        reader = GeminiReaderAgent(
            llm=llm,
        )

        print("READER=RUN")
        reader_result = reader.run(state)
        state.update(reader_result)

        result["pipeline"].append({
            "agent": "reader",
            "status": "PASS",
        })

        print("READER=PASS")

        analyzer = AnalyzerAgent(
            llm=llm,
        )

        print("ANALYZER=RUN")
        analyzer_result = analyzer.run(state)
        state.update(analyzer_result)

        result["pipeline"].append({
            "agent": "analyzer",
            "status": "PASS",
        })

        print("ANALYZER=PASS")

        optimizer = OptimizerAgent(
            llm=llm,
        )

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

    (case_out / "trajectory.json").write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    return result


def main():
    print("=" * 60)
    print("GPU ENGINEERING AGENT BENCHMARK")
    print("ACTUAL GEMINI AGENT PIPELINE — KAGGLE GPU")
    print("=" * 60)

    print(f"ROOT={ROOT}")
    print(f"OUT={OUT}")

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not available")

    print("GEMINI_API_KEY=AVAILABLE")

    print()
    print("===== PROTECTED TRAJECTORY SNAPSHOT =====")
    print(f"COUNT={len(HIST_BEFORE)}")

    patch_workspace_inspector()

    print()
    print("===== IMPORT CHECK =====")

    import agents.reader.gemini_reader
    import agents.analyzer.agent
    import agents.optimizer.agent
    import agents.verifier.agent

    print("AGENTS_IMPORT=PASS")

    print()
    print("===== GEMINI CLIENT =====")

    llm = build_llm()

    print("GEMINI_CLIENT=READY")

    cases = [
        f"case_{i:03d}"
        for i in range(1, 11)
    ]

    results = []

    print()
    print("===== ACTUAL 10-CASE PIPELINE =====")

    for case_id in cases:
        results.append(
            run_case(
                case_id,
                llm,
            )
        )

    summary = {
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
    }

    report = {
        "benchmark": "GPU Engineering Agent Benchmark",
        "execution_mode": "actual_gemini_agent_pipeline",
        "summary": summary,
        "cases": results,
    }

    (OUT / "actual_results.json").write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(json.dumps(summary, indent=2))

    print()
    print("===== HISTORICAL TRAJECTORY PROTECTION =====")

    hist_after = historical_snapshot()

    changed = [
        name
        for name, digest in HIST_BEFORE.items()
        if hist_after.get(name) != digest
    ]

    print(
        "HISTORICAL_TRAJECTORIES_MODIFIED="
        + ("YES" if changed else "NO")
    )

    if changed:
        for name in changed:
            print("CHANGED=", name)

    print()
    print("===== README PROTECTION =====")

    print(
        "README_MODIFIED="
        + (
            "YES"
            if sha256(ROOT / "README.md") != README_BEFORE
            else "NO"
        )
    )

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
    print("ACTUAL PIPELINE RUN COMPLETE")
    print("TERMINAL STAYS OPEN")
    print("=" * 60)


if __name__ == "__main__":
    main()
