# Case 008 — Distributed Gradient Communication Bug

## Objective

Detect and correct an incorrect distributed gradient aggregation boundary.

## Domain

Distributed Training

## Bug

The defective implementation performs an all-reduce sum on the local gradient but applies the summed gradient directly without normalizing it by the distributed world size.

This causes the effective update magnitude to depend on the number of workers.

## Expected Behavior

The distributed gradient must be aggregated deterministically and normalized by the participating world size before it is used for the parameter update.

## Verification

Static contract tests verify:

- distributed initialization guards are present
- gradient communication uses all_reduce
- the reference normalizes the aggregate
- normalization occurs before the parameter update
- the defective implementation preserves the missing normalization defect

## Runtime

The case is designed for Python, PyTorch, and distributed execution environments.

## Trajectory

Agent interaction history is recorded under the trajectory directory when the execution pipeline is used.

## Scope

This case focuses specifically on distributed gradient communication and update scaling.
