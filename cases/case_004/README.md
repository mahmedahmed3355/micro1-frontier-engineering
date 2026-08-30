# Case 004 — CUDA Shared-Memory Synchronization

## Task

Repair a CUDA block-reduction implementation that uses shared memory.

Each CUDA block loads a portion of the input into shared memory and then
performs a tree-style reduction.

The public function returns one partial sum for each CUDA block.

## Failure Mode

The defective implementation updates shared memory across multiple
reduction stages without establishing a synchronization barrier between
those stages.

CUDA threads within a block do not automatically execute each stage in
lockstep.

A thread may therefore read shared memory while another thread is still
updating a value required by the current reduction stage.

## Expected Behavior

The implementation must:

- Correctly load valid input elements into shared memory.
- Treat out-of-range input elements as zero.
- Preserve the reduction structure.
- Synchronize threads whenever the next reduction stage depends on
  shared-memory writes from the previous stage.
- Produce the correct partial sum for every block.
- Return complete results before the public function returns.

## Synchronization Requirements

Shared-memory data dependencies must be explicitly synchronized.

The implementation must not rely on:

- warp scheduling assumptions
- block scheduling order
- instruction timing
- host sleeps
- repeated execution eventually producing the right result

The synchronization strategy must remain correct for arbitrary valid
input sizes.

## Boundary Workloads

Verification should cover:

- one element
- a small partial block
- an exact block
- one element beyond a block
- multiple blocks
- a partially occupied final block
- larger workloads

## Engineering Constraints

Preserve the public run_reduction interface.

Keep the block reduction on the GPU.

Do not replace the computation with a CPU implementation.

Do not serialize the entire workload on the host.

Do not remove shared memory merely to avoid the synchronization issue.

## Evaluation

The verifier will compare every returned block sum against an independent
expected result.

Repeated execution must remain deterministic.

The evaluation should include workloads that exercise multiple reduction
stages and partially occupied blocks.

## Engineering Goal

The objective is to correctly establish shared-memory ordering between
dependent reduction stages while preserving the intended CUDA
parallel-reduction design.
