# Reproduction Guide

## Benchmark Scope

The benchmark contains ten cases:

- case_001
- case_002
- case_003
- case_004
- case_005
- case_006
- case_007
- case_008
- case_009
- case_010

Each case contains its implementation, reference implementation, tests, metadata, and trajectory information.

## Repository Validation

Run the project test suite from the repository root:

python -m pytest -q

Run Ruff validation:

ruff check .

Run formatting validation:

ruff format --check .

## Reference Baselines

Reference baseline tooling is located at:

tools/benchmark/run_reference_baselines.py

Reference benchmark evidence is stored under:

trajectories/benchmark/

The aggregated reference result is:

trajectories/benchmark/reference_baseline_results.json

Per-case baseline records are stored as:

trajectories/benchmark/case_001_baseline.json
through
trajectories/benchmark/case_010_baseline.json

## Candidate Execution

The candidate path uses the Gemini integration.

The Gemini credential is expected through the GEMINI_API_KEY environment variable.

Credentials must never be committed to source code or benchmark artifacts.

## CUDA Execution

CUDA-dependent execution requires a GPU-capable environment.

The benchmark retains reference execution evidence and trajectories needed to inspect and reproduce the benchmark workflow.

## Trajectories

Primary execution trajectories are stored under:

trajectories/execution/

Benchmark-specific evidence is stored under:

trajectories/benchmark/

Reference execution artifacts are stored under:

trajectories/reference_runs/

## Important

README.md is intentionally not modified by this documentation pass.
