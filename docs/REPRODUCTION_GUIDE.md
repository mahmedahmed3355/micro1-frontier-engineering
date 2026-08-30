# Reproduction Guide

## Overview

This repository contains the GPU Engineering Agent Benchmark, its benchmark
runner, case registry, verification tooling, recorded execution evidence, and
canonical final benchmark artifacts.

The finalized benchmark contains 10 cases.

The canonical completed result is:

- Reference baseline: 10/10 PASS
- Gemini candidate: 10/10 PASS
- Candidate success rate: 100%
- Failed cases: none
- Final benchmark report: PASS
- Final case matrix: PASS

## Benchmark Components

The main benchmark components are:

- cases/
- tools/benchmark/run_benchmark.py
- tools/benchmark/run_reference_baselines.py
- tools/benchmark/compare_results.py
- tools/benchmark/reconcile_final_evidence.py
- tools/benchmark/benchmark_contract.json
- tools/batch/case_registry.json
- trajectories/benchmark/final_benchmark_report.json
- trajectories/benchmark/final_case_matrix.json

## Case Registry

The canonical registry is:

tools/batch/case_registry.json

The registry contains:

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

## Local Environment

The benchmark tooling was validated with Python 3.12.

A dedicated local environment can be used for repository-level validation.

The benchmark virtual environment used during this project is:

.venv-benchmark

Before executing any local benchmark command, ensure the required project
dependencies are installed in the selected environment.

## Runner

The benchmark runner is:

tools/benchmark/run_benchmark.py

Inspect its command-line contract with:

.venv-benchmark/bin/python tools/benchmark/run_benchmark.py --help

The runner supports:

--mode dry-run
--mode agent
--case CASE_ID

A dry-run can be used to inspect benchmark selection without performing the
actual agent execution.

Agent mode invokes the existing project Reader, Analyzer, Optimizer, and
Verifier pipeline and requires the configured Gemini client when actual agent
execution is intentionally requested.

## Canonical Final Evidence

The canonical final report is:

trajectories/benchmark/final_benchmark_report.json

The canonical final case matrix is:

trajectories/benchmark/final_case_matrix.json

These artifacts represent the finalized benchmark result.

Expected result:

Cases: 10
Reference baseline: 10/10 PASS
Gemini candidate: 10/10 PASS
Candidate success rate: 100%
Failed cases: none

## Historical Kaggle Execution

The benchmark execution on Kaggle has already been completed for the finalized
checkpoint.

Recorded execution evidence is preserved under:

trajectories/benchmark/actual_kaggle_execution/

The historical execution artifacts and trajectories are evidence and must not
be rewritten merely because later documentation or canonical aggregation was
updated.

## Trajectory Preservation

Historical trajectories are preserved exactly as recorded.

Intermediate historical states such as PENDING or FAILED may appear inside
historical execution records.

Those historical states are not interpreted as the final benchmark result.

The canonical final report and final case matrix provide the final
interpretation of the completed benchmark.

## Evidence Reconciliation

Canonical evidence reconciliation is implemented by:

tools/benchmark/reconcile_final_evidence.py

The reconciliation layer selects the appropriate canonical evidence for each
case while preserving historical intermediate attempts as historical records.

The finalized canonical case results are:

case_001 PASS
case_002 PASS
case_003 PASS
case_004 PASS
case_005 PASS
case_006 PASS
case_007 PASS
case_008 PASS
case_009 PASS
case_010 PASS

## Result Comparison

The result comparator is:

tools/benchmark/compare_results.py

It compares baseline and candidate result objects and reports correctness and
runtime-related metrics.

Execution runtime and CUDA kernel timing are distinct metrics and must not be
mixed.

## Reproduction Validation

Repository-level validation can begin with:

.venv-benchmark/bin/python -m py_compile tools/benchmark/run_benchmark.py tools/benchmark/run_reference_baselines.py tools/benchmark/reconcile_final_evidence.py tools/benchmark/compare_results.py

Then verify:

trajectories/benchmark/final_benchmark_report.json
trajectories/benchmark/final_case_matrix.json
tools/batch/case_registry.json

The expected canonical result is 10/10 baseline and 10/10 candidate.

## Actual Agent Re-execution

Actual Gemini agent execution is separate from reading the finalized
historical evidence.

If an evaluator intentionally re-runs the agent pipeline, the evaluator
must provide the required Gemini credentials and execution environment.

The existing Reader, Analyzer, Optimizer, and Verifier pipeline should be used.

A new execution should not overwrite historical benchmark evidence merely to
replace the recorded results.

## CUDA Re-execution

Cases requiring CUDA must be executed in an environment providing the required
CUDA runtime and compatible toolchain.

CUDA requirements are defined by the case metadata and execution policy.

The completed Kaggle evidence remains the canonical recorded execution for
this benchmark checkpoint.

## Reproduction Checklist

- Python 3.12 environment available
- Case registry contains all 10 cases
- Case directories contain case_001 through case_010
- Benchmark tooling compiles successfully
- Final benchmark report exists
- Final case matrix exists
- Final report status is PASS
- Final case matrix status is PASS
- Reference baseline is 10/10 PASS
- Gemini candidate is 10/10 PASS
- Candidate success rate is 100%
- Failed cases are none
- Historical Kaggle evidence remains preserved
- Historical trajectories remain unchanged

## Current Completion State

Kaggle execution: COMPLETE
Recorded trajectories: PRESERVED
Final benchmark report: PASS
Final case matrix: PASS
Reference baseline: 10/10
Gemini candidate: 10/10
Candidate success rate: 100%
Failed cases: NONE

Future submission deliverables such as the final competition video or PDF
submission document are separate from benchmark execution and reproduction.
