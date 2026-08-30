# Case 002 — CUDA Host/GPU Transfer Correctness

## Task

Repair the CUDA host-to-device and device-to-host transfer implementation.

The function receives host memory, transfers the data to device memory,
and must make the requested result available in host memory before the
function returns.

## Failure Mode

The defective implementation uses an asynchronous device-to-host
transfer while exposing a synchronous function contract to its caller.

The caller is allowed to read host_output immediately after run_transfer
returns.

The implementation must therefore establish the required completion
guarantee before returning.

## Expected Behavior

For every valid positive input size:

- Input data must reach device memory correctly.
- Device-to-host transfer must complete before the function returns.
- host_output must contain the complete expected data on return.
- Device memory must remain valid until all operations using it complete.
- Device memory must be released after use.
- CUDA errors must be propagated to the caller.

## Transfer Scope

The implementation should remain correct for:

- very small transfers
- transfers smaller than a memory page
- transfers around page boundaries
- larger transfers
- repeated transfers
- ordinary pageable host allocations

Pinned host memory may be useful for specific asynchronous transfer
designs, but merely using pageable memory is not itself considered a
failure. The correctness requirement is the completion and lifetime
contract exposed by run_transfer.

## Engineering Constraints

Preserve the public run_transfer interface.

Do not remove the transfer operation.

Do not replace the CUDA implementation with a CPU-only implementation.

Do not rely on accidental synchronization behavior.

Do not change the caller-visible synchronous contract.

The repair must establish a valid relationship between asynchronous work,
memory lifetime, and function return.

## Verification

The verifier must use deterministic input data and multiple transfer
sizes.

Verification must check the complete output rather than a single sample.

Repeated execution must be supported.

The expected result must be independent from the candidate output.

The verifier must distinguish numerical correctness from unsafe
asynchronous lifetime or completion behavior where practical.

## Engineering Goal

The objective is to correctly repair the host/device transfer lifecycle
while preserving the intended CUDA execution model and synchronous
caller contract.
