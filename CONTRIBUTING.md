GPU Engineering Agent Benchmark
Contribution and Execution Contract
====================================

Purpose
-------

This document defines the project-level rules for extending and executing the
GPU Engineering Agent Benchmark.

Case Integrity
--------------

Each benchmark case must remain self-contained.

A case should contain:

- metadata.json
- README.md
- src/
- reference/
- tests/
- trajectory/

The defective implementation and reference implementation must remain
separate.

Reference implementations are used as the baseline for benchmark comparison.

Static Validation
-----------------

Every new or modified case must pass its static contract tests.

Before a case is considered ready, verify:

- case tests pass
- project tests pass
- Ruff passes
- formatting passes
- forbidden artifact checks pass
- Git diff checks pass

Batch Validation
----------------

The case registry is authoritative for batch ordering.

The current benchmark order is:

001
002
003
004
005
006
007
008
009
010

Batch execution must not silently skip or reorder registered cases.

Execution Separation
--------------------

Static validation and GPU execution are separate phases.

Static validation may run without a GPU.

CUDA runtime execution requires a GPU-capable environment.

The local machine must not be treated as a CUDA execution environment when
CUDA-capable hardware is unavailable.

Baseline and Candidate
----------------------

The benchmark compares:

- reference implementation as baseline
- AI-generated implementation as candidate

The baseline must be executed independently from the candidate.

Candidate success must never be inferred from the existence of a reference
implementation.

Verification
------------

A successful candidate must satisfy the case verifier.

Runtime success alone is insufficient.

The verifier is authoritative for correctness.

Benchmark metrics should distinguish:

- correctness
- runtime
- attempts
- failures
- speedup
- trajectory information

Trajectory Recording
--------------------

Agent execution should record sufficient trajectory information to reproduce
and analyze the run.

Trajectory data must not contain API keys, credentials, or other secrets.

Temporary caches and generated Python bytecode must not be treated as
trajectory data.

Gemini API
----------

Gemini API credentials must be provided through environment variables.

The required environment variable is:

GEMINI_API_KEY

Credentials must never be:

- committed
- printed
- embedded in source
- stored in test fixtures
- stored in trajectory files
- stored in benchmark reports

Kaggle Execution
----------------

Kaggle is treated as an execution environment, not as the source of truth for
the benchmark.

The repository and handoff archive must contain everything required to restore
the benchmark.

No case should depend on files created manually inside a temporary Kaggle
session.

Batch execution should be restartable.

If a GPU session terminates, execution must be recoverable from the project
checkpoint and handoff package.

Generated Artifacts
-------------------

Do not commit:

- __pycache__
- *.pyc
- *.pyo
- .pytest_cache
- .ruff_cache
- .mypy_cache
- temporary logs
- API credentials
- temporary GPU-session files

Git Workflow
------------

Create a local Git checkpoint before major execution phases.

Do not push automatically after every change.

Review the complete diff before the final push.

Documentation
-------------

Every case must document:

- the problem
- the intended behavior
- the defective behavior
- the reference behavior
- the verification strategy
- the trajectory purpose

Project-level changes should update the project documentation when the
execution architecture changes.

Difficulty Progression
----------------------

Cases should increase in difficulty incrementally.

Difficulty should be increased through additional interaction complexity,
failure modes, synchronization requirements, memory behavior, distributed
behavior, and verification depth rather than through arbitrary code size.

Existing cases should not be weakened solely to make static tests pass.

Safety and Reproducibility
--------------------------

Do not introduce behavior that depends on:

- undocumented machine state
- local user files
- hard-coded credentials
- non-reproducible external state
- manual intervention during batch execution

All benchmark results should be attributable to the recorded case,
environment, agent execution, and verifier.

Final Benchmark
---------------

case registry → Reference Oracle → Simple Baseline → Advanced Gemini Agent → CUDA execution → deterministic verifier → trajectory recording → baseline vs advanced comparison → aggregate benchmark report
