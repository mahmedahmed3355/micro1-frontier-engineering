# Case 007 — PyTorch DDP Synchronization Bug

## Objective

Detect and correct a distributed-training synchronization defect at the optimization boundary.

## Domain

PyTorch Distributed / DDP

## Bug

The defective implementation performs the local backward pass and reaches the optimizer step without explicitly aggregating the worker-local loss across the distributed workers.

## Expected Behavior

When distributed execution is initialized, workers must coordinate before crossing the optimization boundary and the reference loss must be aggregated deterministically across the participating workers.

## Verification

Static contract tests verify:

- distributed APIs are present
- synchronization occurs after backward
- aggregation occurs before optimizer.step
- the defective implementation lacks the required aggregation
- the reference performs synchronization and aggregation

## Runtime

The case is designed for Python, PyTorch, and distributed execution environments.

## Trajectory

Agent interaction history is recorded under the trajectory directory when the execution pipeline is used.

## Scope

This case focuses specifically on distributed synchronization and communication ordering around a PyTorch optimization step.
