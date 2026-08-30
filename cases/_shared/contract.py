from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaseContract:
    case_id: str
    title: str
    domain: str
    bug_class: str
    primary_failure: str
    expected_behavior: str
    verification_strategy: str
    required_runtime: tuple[str, ...]


CASE_CONTRACTS: tuple[CaseContract, ...] = (
    CaseContract(
        case_id="001",
        title="CUDA Kernel Indexing Boundary Bug",
        domain="CUDA Kernel",
        bug_class="indexing / boundary conditions",
        primary_failure="invalid thread indexing or missing bounds guard",
        expected_behavior="every valid element is processed exactly once",
        verification_strategy="boundary sizes including non-multiples of block size",
        required_runtime=("cuda",),
    ),
    CaseContract(
        case_id="002",
        title="CUDA Host GPU Transfer Bug",
        domain="CUDA Memory",
        bug_class="host/device transfer ordering or pinned/pageable handling",
        primary_failure="incorrect transfer visibility or synchronization",
        expected_behavior="transfers complete with correct data visibility",
        verification_strategy="transfer correctness plus synchronization checks",
        required_runtime=("cuda",),
    ),
    CaseContract(
        case_id="003",
        title="CUDA Stream Event Dependency Bug",
        domain="CUDA Streams",
        bug_class="incorrect stream/event dependency",
        primary_failure="consumer executes before producer completion",
        expected_behavior="consumer observes producer results only after completion",
        verification_strategy="repeated dependency-ordering execution",
        required_runtime=("cuda",),
    ),
    CaseContract(
        case_id="004",
        title="CUDA Synchronization Race Bug",
        domain="CUDA Synchronization",
        bug_class="shared-memory or thread synchronization race",
        primary_failure="cooperating threads observe stale shared state",
        expected_behavior="cooperating threads observe synchronized state",
        verification_strategy="repeated race-sensitive deterministic verification",
        required_runtime=("cuda",),
    ),
    CaseContract(
        case_id="005",
        title="CUDA Memory Lifetime Bug",
        domain="CUDA Memory Lifetime",
        bug_class="invalid allocation lifetime or leaked device memory",
        primary_failure="memory is released too early or not released",
        expected_behavior="allocations remain valid for their complete use lifetime",
        verification_strategy="allocation lifecycle and repeated execution checks",
        required_runtime=("cuda",),
    ),
    CaseContract(
        case_id="006",
        title="PyTorch CUDA Device Mismatch",
        domain="PyTorch + CUDA",
        bug_class="CPU/GPU or cross-device tensor mismatch",
        primary_failure="operation combines incompatible devices",
        expected_behavior=(
            "all participating tensors and modules use compatible devices"
        ),
        verification_strategy="device consistency and computation checks",
        required_runtime=("python", "torch", "cuda"),
    ),
    CaseContract(
        case_id="007",
        title="PyTorch DDP Synchronization Bug",
        domain="PyTorch DDP",
        bug_class="worker synchronization / collective mismatch",
        primary_failure="workers reach inconsistent collective states",
        expected_behavior="workers reach required synchronization points consistently",
        verification_strategy="multi-process state and collective verification",
        required_runtime=("python", "torch", "distributed"),
    ),
    CaseContract(
        case_id="008",
        title="Distributed Gradient Communication Bug",
        domain="Distributed Training",
        bug_class="incorrect accumulation or communication boundary",
        primary_failure=(
            "gradient communication occurs at the wrong optimization boundary"
        ),
        expected_behavior=(
            "effective gradients match the intended optimization boundary"
        ),
        verification_strategy="controlled accumulation versus reference update",
        required_runtime=("python", "torch", "distributed"),
    ),
    CaseContract(
        case_id="009",
        title="Async Backend GPU Worker Bug",
        domain="Backend + GPU",
        bug_class="incorrect asynchronous worker lifecycle or synchronization",
        primary_failure="GPU work is lost, duplicated, or finalized prematurely",
        expected_behavior="requests complete with valid worker lifecycle",
        verification_strategy="concurrent request and worker lifecycle verification",
        required_runtime=("python", "cuda", "async"),
    ),
    CaseContract(
        case_id="010",
        title="Full System Distributed GPU Configuration Bug",
        domain="Backend + CUDA + Distributed + Configuration",
        bug_class="compound configuration/orchestration failure",
        primary_failure="configuration is lost across execution boundaries",
        expected_behavior=(
            "configuration propagates through the complete execution path"
        ),
        verification_strategy="end-to-end deterministic system verification",
        required_runtime=("python", "cuda", "torch", "distributed", "async"),
    ),
)


def get_contract(case_id: str) -> CaseContract:
    for contract in CASE_CONTRACTS:
        if contract.case_id == case_id:
            return contract
    raise KeyError(f"Unknown case: {case_id}")


def case_root(root: Path, case_id: str) -> Path:
    return root / "cases" / f"case_{case_id}"
