# Case 006 — PyTorch CUDA Device Mismatch

## Objective

Detect and correct a PyTorch CUDA device mismatch where participating tensors are not moved to the same execution device before the computation.

## Domain

PyTorch / CUDA

## Bug

The defective implementation performs a tensor operation without guaranteeing that all participating tensors use the same CUDA device.

## Expected Behavior

The implementation must select the intended execution device and move all participating tensors and model parameters required by the operation to that compatible device before computation.

## Verification

The case uses deterministic static contract checks to verify:

- device selection is explicit
- participating tensors are moved to the selected device
- the computation occurs after device placement
- the defective implementation preserves the intended mismatch
- the reference implementation establishes device consistency

## Runtime

The case is designed for Python and PyTorch CUDA environments.

## Trajectory

Agent interaction history is recorded under the case trajectory directory when the execution pipeline is used.

## Scope

This case focuses specifically on device consistency at the PyTorch computation boundary.
