# CASE 009 runtime verification runner
# Corrected entrypoint: run_gpu_worker

import asyncio
import importlib.util
import sys
from pathlib import Path

import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise RuntimeError(f"cannot load {path}")

    spec.loader.exec_module(module)
    return module


async def execute_worker(module, use_cuda: bool):
    device = torch.device("cuda:0" if use_cuda else "cpu")

    async def work():
        if use_cuda:
            x = torch.arange(
                4096,
                dtype=torch.float32,
                device=device,
            )

            y = x * 3.0 + 7.0

            for _ in range(100):
                y = y * 1.000001 + 0.000001

            return y

        return torch.arange(
            4096,
            dtype=torch.float32,
            device=device,
        )

    return await module.run_gpu_worker(work)


def expected_cuda_result():
    result = torch.arange(
        4096,
        dtype=torch.float32,
        device="cpu",
    )

    result = result * 3.0 + 7.0

    for _ in range(100):
        result = result * 1.000001 + 0.000001

    return result


async def main(source_path: str, reference_path: str):
    candidate = load_module(
        "case009_candidate",
        Path(source_path),
    )

    reference = load_module(
        "case009_reference",
        Path(reference_path),
    )

    results = []

    candidate_cpu = await execute_worker(candidate, False)
    reference_cpu = await execute_worker(reference, False)

    cpu_equal = torch.equal(
        candidate_cpu,
        reference_cpu,
    )

    results.append({
        "test": "CPU_ASYNC_RESULT",
        "candidate_device": str(candidate_cpu.device),
        "reference_device": str(reference_cpu.device),
        "status": "PASS" if cpu_equal else "FAIL",
    })

    if torch.cuda.is_available():

        candidate_gpu = await execute_worker(candidate, True)
        reference_gpu = await execute_worker(reference, True)

        expected = expected_cuda_result()

        candidate_host = candidate_gpu.cpu()
        reference_host = reference_gpu.cpu()

        candidate_correct = torch.allclose(
            candidate_host,
            expected,
            rtol=1e-5,
            atol=1e-5,
        )

        reference_correct = torch.allclose(
            reference_host,
            expected,
            rtol=1e-5,
            atol=1e-5,
        )

        implementations_match = torch.allclose(
            candidate_host,
            reference_host,
            rtol=1e-5,
            atol=1e-5,
        )

        results.append({
            "test": "CUDA_ASYNC_RESULT",
            "candidate_device": str(candidate_gpu.device),
            "reference_device": str(reference_gpu.device),
            "candidate_correct": bool(candidate_correct),
            "reference_correct": bool(reference_correct),
            "candidate_matches_reference": bool(implementations_match),
            "status": (
                "PASS"
                if candidate_correct
                and reference_correct
                and implementations_match
                else "FAIL"
            ),
        })

        repeated_ok = True

        for iteration in range(10):
            candidate_result = await execute_worker(
                candidate,
                True,
            )

            reference_result = await execute_worker(
                reference,
                True,
            )

            candidate_host = candidate_result.cpu()
            reference_host = reference_result.cpu()

            if not torch.allclose(
                candidate_host,
                reference_host,
                rtol=1e-5,
                atol=1e-5,
            ):
                repeated_ok = False
                print(
                    f"REPEATED_FAILURE_ITERATION={iteration}"
                )
                break

        results.append({
            "test": "CUDA_REPEATED_ASYNC_EXECUTION",
            "iterations": 10,
            "status": "PASS" if repeated_ok else "FAIL",
        })

    else:
        results.append({
            "test": "CUDA_ASYNC_RESULT",
            "status": "SKIP",
            "reason": "CUDA unavailable",
        })

        results.append({
            "test": "CUDA_REPEATED_ASYNC_EXECUTION",
            "status": "SKIP",
            "reason": "CUDA unavailable",
        })

    return results


if __name__ == "__main__":
    results = asyncio.run(
        main(
            sys.argv[1],
            sys.argv[2],
        )
    )

    print("RESULTS_BEGIN")

    for result in results:
        print(result)

    print("RESULTS_END")

    failures = [
        result
        for result in results
        if result["status"] == "FAIL"
    ]

    print("TOTAL=", len(results))
    print("FAILED=", len(failures))
    print(
        "OVERALL_STATUS=",
        "PASS" if not failures else "FAIL",
    )

    raise SystemExit(
        0 if not failures else 10
    )
