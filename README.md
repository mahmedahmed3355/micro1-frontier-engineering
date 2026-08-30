GPU Engineering Agent Benchmark
===============================

Overview
--------

GPU Engineering Agent Benchmark is a benchmark for evaluating AI agents on GPU
engineering and CUDA-oriented debugging tasks.

The benchmark is organized as a collection of independent cases. Each case
contains a defective implementation, a reference implementation, static
contract tests, documentation, and trajectory storage.

Current Scope
-------------

The current benchmark contains ten cases:

case_001

case_002

case_003

case_004

case_005

case_006

case_007

case_008

case_009

case_010

The cases are intentionally incremental in difficulty. Later cases introduce
more complex interactions involving synchronization, memory lifetime, device
placement, distributed execution, and GPU-oriented infrastructure behavior.

Case Structure
--------------

Each case follows the benchmark case structure:

cases/case_XXX/

├── metadata.json

├── README.md

├── src/

├── reference/

├── tests/

└── trajectory/

The src directory contains the defective implementation.

The reference directory contains the expected correct implementation.

The tests directory contains the static contract and verification tests.

The trajectory directory is reserved for execution and agent trajectory data.

Validation
----------

Static validation is performed before GPU execution.

The current project validates:

- case structure
- static contracts
- reference implementations
- defective implementations
- project-wide tests
- Ruff checks
- formatting
- forbidden generated artifacts
- batch registry consistency

The current static batch contains ten cases and has been validated successfully.

Batch Infrastructure
---------------------

Batch infrastructure is located under:

tools/batch/

The batch layer provides:

- case registry validation
- static batch execution
- result contracts
- batch contract tests

Benchmark Infrastructure
-------------------------

Benchmark-specific contracts are located under:

tools/benchmark/

The benchmark comparison is designed around two execution paths:

1. Reference baseline
2. Gemini Flash agent candidate

The final benchmark will compare the candidate against the reference baseline
using correctness, runtime, speedup, attempts, failures, and trajectory data.

Execution Model
---------------

The intended execution flow is:

1. Validate the case registry.
2. Execute the reference baseline.
3. Execute the AI-generated candidate.
4. Run the verifier.
5. Record execution and trajectory data.
6. Compare candidate results with the baseline.
7. Aggregate results across all cases.

CUDA execution is deferred to a GPU-capable environment.

The local development environment is used for case construction, static
validation, orchestration development, and benchmark preparation.

GPU execution will be performed in a suitable GPU environment such as Kaggle
when the execution phase begins.

Gemini API
----------

Gemini Flash integration is part of the candidate-agent execution phase.

API credentials must not be committed to the repository.

The API key should be supplied through an environment variable:

GEMINI_API_KEY

No API key should appear in source files, test files, trajectories, reports,
or Git history.

Reproducibility
---------------

The benchmark is designed to support batch execution and recovery from
temporary execution environments.

The project prepares a handoff archive containing the cases, batch
infrastructure, static evidence, manifest information, and checksums before
GPU execution.

This allows the benchmark to be restored if a temporary GPU session expires.

Git Workflow
------------

Local checkpoints are created before major execution phases.

Pushes to the remote repository are intentionally deferred until the relevant
execution and benchmark artifacts have been reviewed.

Generated caches and temporary execution artifacts must not be committed.

Status
------

Current preparation status:

Cases 001-010:
READY

Static batch:
10/10 PASS

Project tests:
PASS

Ruff:
PASS

Formatting:
PASS

Forbidden artifacts:
NONE

CUDA execution:
PENDING

Gemini execution:
PENDING

Kaggle execution:
PENDING

Final benchmark report:
PENDING

License
-------

This project is distributed under the MIT License. See LICENSE for the full
license text.

## Agent Architecture

The benchmark includes a multi-agent engineering pipeline built around LangGraph.

### Agents

The current agent layer contains five functional Agent classes:

- ReaderAgent — reads and grounds the engineering workspace so downstream agents can reason from the actual project state.
- AnalyzerAgent — analyzes the discovered implementation and identifies the relevant engineering issue and reasoning path.
- OptimizerAgent — produces the corrective implementation strategy and applies the optimization/fix workflow.
- VerifierAgent — performs deterministic verification of the resulting implementation against the benchmark's expected contract.
- GeminiReaderAgent — specialized Gemini-backed reader used for Gemini-based workspace/code understanding and grounding.

`BaseAgent` is the shared base class and is not counted as an independent functional agent.

### LangGraph Orchestration

The main LangGraph workflow is implemented as a stateful engineering pipeline:

ReaderAgent → AnalyzerAgent → OptimizerAgent → VerifierAgent

The graph also contains explicit retry and exhausted control-flow nodes. A failed verification can route execution back to the optimizer for another attempt, while the exhausted path terminates the workflow.

The orchestration layer therefore evaluates more than a single model response: it preserves agent state, maintains the engineering workflow order, performs verification, and supports controlled retries.

### Agent Execution Order

The project tests explicitly verify the LLM interaction order:

1. Reader
2. Analyzer
3. Optimizer

The verifier then evaluates the resulting implementation, with retry control returning to the optimizer when required.

### Gemini Integration

`GeminiReaderAgent` provides the Gemini-specific reading/grounding path. The benchmark currently keeps Gemini API execution separate from the static validation layer.

At the current benchmark checkpoint:

- Static validation: complete
- Cases 001–010: complete
- LangGraph orchestration: implemented
- Agent architecture: implemented
- Gemini API execution: pending
- CUDA execution: pending
- Kaggle execution: pending

This separation allows the benchmark to validate its contracts and orchestration locally before consuming GPU/Kaggle execution time.
