# Case 010 — Compound Distributed GPU Configuration Bug

## Objective

Diagnose and correct a compound distributed GPU execution problem spanning worker configuration, CUDA device selection, distributed initialization, collective input preparation, and GPU completion.

## Domain

Distributed GPU Infrastructure

## Bug

The defective implementation uses a generic CUDA device instead of deriving the worker-local device from rank.

The execution path additionally omits explicit CUDA synchronization after the distributed collective.

These defects interact across configuration and execution boundaries.

## Expected Behavior

Each worker must derive its CUDA device from its rank.

The selected device must propagate through worker initialization and tensor placement.

Distributed collective input must be contiguous, and GPU execution must reach an explicit completion boundary before the result is returned.

## Verification

Static contract tests verify:

- rank-derived CUDA device construction
- worker configuration propagation
- NCCL initialization
- CUDA device assignment
- CUDA tensor placement
- contiguous collective input
- all-reduce execution
- post-collective CUDA synchronization
- correct ordering of the complete execution path
- preservation of the defective configuration in the source implementation

## Difficulty

This is the highest-difficulty case in the initial 001-010 progression.

It intentionally combines multiple boundaries rather than testing a single isolated CUDA mistake.

## Trajectory

Execution trajectories are stored under the trajectory directory.

## Runtime

The intended runtime includes Python, PyTorch, CUDA, and distributed execution support.

Static validation does not require an active multi-GPU environment.

## Progression

001-004 cover isolated CUDA correctness and synchronization failures.

005 covers CUDA memory lifetime.

006 covers PyTorch device consistency.

007-008 cover distributed training and gradient communication.

009 introduces asynchronous GPU worker coordination.

010 combines configuration, distributed initialization, tensor preparation, collective execution, and GPU completion into one end-to-end failure surface.
