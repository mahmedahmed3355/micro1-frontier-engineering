from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist


ROOT = Path("/kaggle/working/micro1-frontier-engineering")
CASE_ROOT = ROOT / "cases" / "case_010"

SOURCE = CASE_ROOT / "src" / "system.py"
REFERENCE = CASE_ROOT / "reference" / "system.py"
README = CASE_ROOT / "README.md"
TRAJECTORY = ROOT / "trajectories" / "execution" / "case_010.json"

EVIDENCE_DIR = (
    ROOT
    / "trajectories"
    / "benchmark"
    / "actual_kaggle_execution"
    / "case_010"
)

RUNTIME_EVIDENCE = EVIDENCE_DIR / "runtime_verification.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def main() -> int:
    print("=" * 70)
    print("CASE 010 — RUNTIME VERIFICATION")
    print("=" * 70)

    protected = {
        "trajectory": TRAJECTORY,
        "readme": README,
        "source": SOURCE,
        "reference": REFERENCE,
    }

    print("\n===== PRE-CHECK =====")

    for name, path in protected.items():
        print(f"{name.upper()}=", "FOUND" if path.is_file() else "MISSING")

    if not all(path.is_file() for path in protected.values()):
        print("PRE_CHECK=FAIL")
        return 10

    before = {
        name: sha256(path)
        for name, path in protected.items()
    }

    print("\n===== PROTECTED HASHES BEFORE =====")

    for name, digest in before.items():
        print(digest, protected[name].relative_to(ROOT))

    print("\n===== ENVIRONMENT =====")
    print("TORCH_VERSION=", torch.__version__)
    print("CUDA_AVAILABLE=", torch.cuda.is_available())
    print("CUDA_DEVICE_COUNT=", torch.cuda.device_count())
    print("DISTRIBUTED_AVAILABLE=", dist.is_available())

    if not torch.cuda.is_available():
        print("CUDA_ENVIRONMENT=FAIL")
        return 20

    if torch.cuda.device_count() < 2:
        print("CUDA_DEVICE_COUNT_REQUIREMENT=FAIL")
        return 21

    for index in range(torch.cuda.device_count()):
        print(
            f"CUDA_DEVICE_{index}=",
            torch.cuda.get_device_name(index),
        )

    print("ENVIRONMENT_CHECK=PASS")

    print("\n===== ENTRYPOINT DISCOVERY =====")

    candidate = load_module(
        SOURCE,
        "case010_candidate_runtime",
    )

    reference = load_module(
        REFERENCE,
        "case010_reference_runtime",
    )

    required = (
        "WorkerConfig",
        "build_worker_config",
        "initialize_worker",
        "run_step",
    )

    for label, module in (
        ("CANDIDATE", candidate),
        ("REFERENCE", reference),
    ):
        functions = [
            name
            for name in dir(module)
            if not name.startswith("_")
            and callable(getattr(module, name))
        ]

        print(f"{label} FUNCTIONS=", functions)

        for name in required:
            if not hasattr(module, name):
                print(f"{label}_{name}=MISSING")
                return 30

        print(f"{label}_API=PASS")

    print("\n===== CONFIGURATION CONTRACT =====")

    source_text = SOURCE.read_text(encoding="utf-8")
    reference_text = REFERENCE.read_text(encoding="utf-8")

    candidate_generic_cuda = 'device="cuda"' in source_text
    reference_rank_device = 'device = f"cuda:{rank}"' in reference_text

    print(
        "CANDIDATE_GENERIC_CUDA=",
        candidate_generic_cuda,
    )
    print(
        "REFERENCE_RANK_DERIVED_DEVICE=",
        reference_rank_device,
    )

    if not candidate_generic_cuda:
        print("CANDIDATE_DEFECT_PRESERVED=NO")
        return 31

    if not reference_rank_device:
        print("REFERENCE_RANK_DEVICE=NO")
        return 32

    print("CONFIGURATION_CONTRACT=PASS")

    print("\n===== WORKER CONFIGURATION =====")

    candidate_config = candidate.build_worker_config(
        rank=0,
        world_size=1,
    )

    reference_config = reference.build_worker_config(
        rank=0,
        world_size=1,
    )

    print("CANDIDATE_CONFIG=", candidate_config)
    print("REFERENCE_CONFIG=", reference_config)

    if str(candidate_config.device) != "cuda":
        print("CANDIDATE_GENERIC_DEVICE=FAIL")
        return 40

    if str(reference_config.device) != "cuda:0":
        print("REFERENCE_RANK_DEVICE=FAIL")
        return 41

    print("CANDIDATE_DEFECT_CONFIRMED=YES")
    print("REFERENCE_RANK_DEVICE_CONFIRMED=YES")

    result = {
        "case_id": "case_010",
        "verification": "runtime",
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "distributed_available": dist.is_available(),
        },
        "candidate": {
            "generic_cuda_device": candidate_generic_cuda,
            "config_device": str(candidate_config.device),
            "defect_confirmed": True,
        },
        "reference": {
            "rank_derived_device": reference_rank_device,
            "config_device": str(reference_config.device),
            "rank_device_confirmed": True,
        },
        "status": "PASS",
    }

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    RUNTIME_EVIDENCE.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n===== SAVED EVIDENCE =====")
    print(RUNTIME_EVIDENCE.relative_to(ROOT))

    print("\n===== PROTECTED HASHES AFTER =====")

    after = {
        name: sha256(path)
        for name, path in protected.items()
    }

    for name, digest in after.items():
        print(digest, protected[name].relative_to(ROOT))

    if before != after:
        print("PROTECTED_FILES_UNCHANGED=NO")
        return 50

    print("PROTECTED_FILES_UNCHANGED=YES")

    print("\n===== SAFETY =====")
    print("CASE_010_ONLY=YES")
    print("GEMINI_API_CALL=NOT_REQUIRED")
    print("SOURCE_MODIFICATION=NO")
    print("REFERENCE_MODIFICATION=NO")
    print("HISTORICAL_TRAJECTORY_OVERWRITE=NO")
    print("README_MODIFIED=NO")
    print("GIT_OPERATION=NONE")

    print("\n" + "=" * 70)
    print("CASE_010_RUNTIME=PASS")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
