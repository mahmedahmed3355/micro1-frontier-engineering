# Case 009 — Async Backend GPU Worker Bug

## Objective

Detect and correct a missing GPU completion boundary in an asynchronous backend worker.

## Domain

Async Backend / GPU Execution

## Bug

The defective worker awaits the asynchronous Python task but does not explicitly synchronize the CUDA device before moving a CUDA result to host memory.

This creates an incorrect assumption that asynchronous task completion is equivalent to GPU execution completion.

## Expected Behavior

The worker must await the asynchronous operation and, when the result resides on CUDA, synchronize the relevant CUDA device before performing the host-side transfer.

## Verification

Static contract tests verify:

- asynchronous task creation
- task awaiting
- CUDA-result detection
- device synchronization
- synchronization ordering relative to CPU transfer
- preservation of the missing synchronization defect in the defective implementation

## Runtime

The case is designed for Python, PyTorch, asyncio, and CUDA-capable execution environments.

## Trajectory

Agent interaction history is recorded under the trajectory directory when the execution pipeline is used.

## Scope

This case focuses on the boundary between asynchronous backend scheduling and CUDA execution completion.

## Difficulty Progression

This case is intentionally more difficult than the earlier isolated CUDA and distributed cases because it combines asynchronous control flow with GPU execution semantics.

Case 010 is reserved for the compound end-to-end distributed GPU configuration problem.
