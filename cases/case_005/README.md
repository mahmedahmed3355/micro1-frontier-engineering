# Case 005 — Adversarial CUDA Memory Lifetime Vector

## Objective

Detect an incomplete cleanup path in a CUDA host/device memory lifecycle.

## Defect

The defective implementation allocates both device input and device output buffers.

Failure paths clean up allocations that are already owned.

The adversarial defect is on the successful execution path: the implementation frees the input allocation but does not release the output allocation before returning.

The important distinction is that the output cleanup exists on error paths, but is absent from the successful completion path.

This prevents a simplistic search for `cudaFree(device_output)` from being sufficient.

## Expected Behavior

The successful path must release both:

1. `device_output`
2. `device_input`

Cleanup must occur after the final device-to-host operation and before the function returns success.

Cleanup failures must remain observable.

## Adversarial Vector

The case deliberately contains cleanup of `device_output` on earlier error paths.

An agent must therefore reason about control flow rather than conclude that the allocation is safe merely because a `cudaFree(device_output)` call exists somewhere in the function.

## Verification

Static tests validate:

- both device allocations
- host-to-device transfer
- device-to-host transfer
- cleanup on allocation/copy failure paths
- missing successful-path output cleanup in the defective source
- output cleanup in the reference success path
- input cleanup in the reference success path
- cleanup ordering before the final return

## Runtime

The case targets CUDA runtime semantics.

Static validation does not require actual CUDA execution.

## Trajectory

Execution trajectory artifacts belong under `trajectory/`.

The trajectory should capture the agent's observations, reasoning-relevant actions, modifications, verification results, and rejected approaches during benchmark execution.

## Difficulty

This is an adversarial memory-lifetime vector positioned between the basic CUDA cases and the higher-level PyTorch/distributed cases.

It increases difficulty through control-flow ambiguity rather than introducing unrelated APIs.
