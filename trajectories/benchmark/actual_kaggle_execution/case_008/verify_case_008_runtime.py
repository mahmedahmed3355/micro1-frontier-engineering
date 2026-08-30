from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist


ROOT = Path("/kaggle/working/micro1-frontier-engineering")
SOURCE = ROOT / "cases/case_008/src/optimizer.py"
REFERENCE = ROOT / "cases/case_008/reference/optimizer.py"
OUTDIR = ROOT / "trajectories/benchmark/actual_kaggle_execution/case_008"
OUTDIR.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def load_module(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(rank: int, world_size: int, port: int, queue) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)

        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
        )

        candidate = load_module(
            SOURCE,
            f"case008_candidate_{rank}",
        )
        reference = load_module(
            REFERENCE,
            f"case008_reference_{rank}",
        )

        # Rank-local gradients: rank 0 -> 1.0, rank 1 -> 2.0
        gradient_value = float(rank + 1)

        candidate_parameter = torch.tensor(10.0)
        reference_parameter = torch.tensor(10.0)

        gradient = torch.tensor(gradient_value)

        candidate.distributed_update(
            candidate_parameter,
            gradient,
            1.0,
        )

        reference.distributed_update(
            reference_parameter,
            gradient,
            1.0,
        )

        # Every rank should observe the same final parameter.
        candidate_gather = [
            torch.zeros_like(candidate_parameter)
            for _ in range(world_size)
        ]
        reference_gather = [
            torch.zeros_like(reference_parameter)
            for _ in range(world_size)
        ]

        dist.all_gather(
            candidate_gather,
            candidate_parameter,
        )
        dist.all_gather(
            reference_gather,
            reference_parameter,
        )

        queue.put(
            {
                "rank": rank,
                "candidate_parameter": float(
                    candidate_parameter.item()
                ),
                "reference_parameter": float(
                    reference_parameter.item()
                ),
                "candidate_all_ranks": [
                    float(x.item()) for x in candidate_gather
                ],
                "reference_all_ranks": [
                    float(x.item()) for x in reference_gather
                ],
            }
        )

        dist.barrier()
        dist.destroy_process_group()

    except Exception as exc:
        queue.put(
            {
                "rank": rank,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def main() -> int:
    print("=" * 70)
    print("CASE 008 — DISTRIBUTED RUNTIME VERIFICATION")
    print("=" * 70)

    print("\n===== PROTECTED HASHES BEFORE =====")

    protected = {
        "trajectory": ROOT / "trajectories/execution/case_008.json",
        "readme": ROOT / "cases/case_008/README.md",
        "source": SOURCE,
        "reference": REFERENCE,
    }

    before = {}

    for name, path in protected.items():
        digest = sha256(path)
        before[name] = digest
        print(digest, path.relative_to(ROOT))

    print("\n===== SOURCE / REFERENCE CONTRACT =====")

    source_text = SOURCE.read_text(encoding="utf-8")
    reference_text = REFERENCE.read_text(encoding="utf-8")

    print(
        "CANDIDATE_ALL_REDUCE=",
        "dist.all_reduce" in source_text,
    )
    print(
        "CANDIDATE_NORMALIZATION=",
        "local_gradient.div_(dist.get_world_size())"
        in source_text,
    )
    print(
        "REFERENCE_ALL_REDUCE=",
        "dist.all_reduce" in reference_text,
    )
    print(
        "REFERENCE_NORMALIZATION=",
        "local_gradient.div_(dist.get_world_size())"
        in reference_text,
    )

    print("\n===== ENVIRONMENT =====")
    print("TORCH_VERSION=", torch.__version__)
    print("DISTRIBUTED_AVAILABLE=", dist.is_available())

    if not dist.is_available():
        print("OVERALL_STATUS=FAIL")
        return 2

    print("\n===== DISTRIBUTED EXECUTION =====")

    world_size = 2
    port = free_port()

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    processes = [
        ctx.Process(
            target=worker,
            args=(rank, world_size, port, queue),
        )
        for rank in range(world_size)
    ]

    for process in processes:
        process.start()

    results = []

    try:
        for _ in range(world_size):
            results.append(queue.get(timeout=60))
    finally:
        for process in processes:
            process.join(timeout=60)

    print("\n===== RUNTIME RESULTS =====")

    for result in sorted(results, key=lambda x: x["rank"]):
        print(json.dumps(result, indent=2))

    if any("error" in result for result in results):
        print("\nOVERALL_STATUS=FAIL")
        return 3

    # rank 0 gradient = 1
    # rank 1 gradient = 2
    #
    # SUM  = 3
    # MEAN = 1.5
    #
    # Candidate defect:
    #   10 - 3.0 = 7.0
    #
    # Reference:
    #   10 - 1.5 = 8.5

    expected_candidate = 7.0
    expected_reference = 8.5

    candidate_values = [
        result["candidate_parameter"]
        for result in results
    ]

    reference_values = [
        result["reference_parameter"]
        for result in results
    ]

    candidate_defect_confirmed = all(
        abs(value - expected_candidate) < 1e-6
        for value in candidate_values
    )

    reference_correct = all(
        abs(value - expected_reference) < 1e-6
        for value in reference_values
    )

    candidate_differs_from_reference = all(
        abs(c - r) > 1e-6
        for c, r in zip(candidate_values, reference_values)
    )

    print("\n===== ASSERTIONS =====")
    print(
        "CANDIDATE_DEFECT_CONFIRMED=",
        candidate_defect_confirmed,
    )
    print(
        "REFERENCE_NORMALIZATION_CONFIRMED=",
        reference_correct,
    )
    print(
        "CANDIDATE_DIFFERS_FROM_REFERENCE=",
        candidate_differs_from_reference,
    )

    print("\n===== PROTECTED HASHES AFTER =====")

    unchanged = True

    for name, path in protected.items():
        digest = sha256(path)
        same = digest == before[name]
        unchanged = unchanged and same

        print(
            digest,
            path.relative_to(ROOT),
            "UNCHANGED=",
            same,
        )

    overall = (
        candidate_defect_confirmed
        and reference_correct
        and candidate_differs_from_reference
        and unchanged
    )

    result = {
        "case_id": "case_008",
        "verification": "distributed_runtime",
        "status": "PASS" if overall else "FAIL",
        "world_size": world_size,
        "torch_version": torch.__version__,
        "candidate_defect_confirmed": candidate_defect_confirmed,
        "reference_normalization_confirmed": reference_correct,
        "candidate_differs_from_reference":
            candidate_differs_from_reference,
        "expected_candidate_parameter": expected_candidate,
        "expected_reference_parameter": expected_reference,
        "candidate_parameters": candidate_values,
        "reference_parameters": reference_values,
        "protected_files_unchanged": unchanged,
        "protected_hashes_before": before,
    }

    output = OUTDIR / "runtime_verification.json"
    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n===== SAFETY =====")
    print("CASE_008_ONLY=YES")
    print("GEMINI_API_CALL=NOT_REQUIRED")
    print("CUDA_EXECUTION=NOT_REQUIRED")
    print("SOURCE_MODIFICATION=NO")
    print("REFERENCE_MODIFICATION=NO")
    print("HISTORICAL_TRAJECTORY_MODIFICATION=NO")
    print("GIT_OPERATION=NONE")
    print("PROTECTED_FILES_UNCHANGED=", unchanged)

    print("\n===== SAVED EVIDENCE =====")
    print(output.relative_to(ROOT))

    print("\n" + "=" * 70)
    print("CASE_008_RUNTIME=", "PASS" if overall else "FAIL")
    print("=" * 70)

    return 0 if overall else 5


if __name__ == "__main__":
    main()
