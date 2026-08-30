# Case 001 — CUDA Kernel Indexing and Boundary Conditions

## Task

Repair the CUDA vector-add implementation in this case.

The implementation computes one output element per CUDA thread. The
workload size is not guaranteed to be an exact multiple of the CUDA block
size.

The current implementation has an indexing and boundary-handling defect
that can cause threads outside the logical input range to access memory
that does not belong to the requested workload.

## Expected Behavior

The repaired implementation must:

- Produce the correct vector-add result for every valid input element.
- Correctly handle workloads smaller than one CUDA block.
- Correctly handle workloads exactly equal to one CUDA block.
- Correctly handle workloads larger than one block.
- Correctly handle workloads whose size is not divisible by the block size.
- Avoid invalid memory accesses caused by out-of-range thread indices.
- Preserve the existing public launcher interface.

## Constraints

Do not change the mathematical operation.

Do not remove valid work from the computation.

Do not solve the problem by assuming that the input size is always a
multiple of the CUDA block size.

Do not change the caller-visible interface.

The solution must remain correct for arbitrary positive input sizes.

## Investigation Guidance

Inspect the CUDA kernel indexing logic and compare the logical workload
size with the number of launched threads.

Pay particular attention to the final partially occupied CUDA block.

The launch configuration may intentionally contain more threads than
logical elements. The kernel is responsible for ensuring that each
thread accesses only a valid logical element.

## Correctness Requirements

For an input size N, valid element indices are in the range from zero
through N minus one.

Every valid output element must contain the sum of the corresponding
input elements.

Threads whose calculated global index is outside the logical workload
must not access the input or output arrays.

## Boundary Workloads

The implementation must correctly handle at least the following classes
of workload sizes:

- A single element.
- A workload smaller than the CUDA block size.
- A workload exactly equal to the CUDA block size.
- A workload one element larger than the CUDA block size.
- Multiple CUDA blocks.
- A workload that ends partway through the final CUDA block.
- A larger non-multiple-of-block-size workload.

## Evaluation

Evaluation will compare the candidate implementation against the
required behavior using deterministic correctness checks and boundary
focused workloads.

Passing a single convenient input size is not sufficient.

The implementation must generalize to the complete valid input range
covered by the benchmark.

## Engineering Goal

The objective is not merely to make one test pass.

The objective is to identify and correctly repair the kernel's global
thread indexing boundary condition while preserving the intended CUDA
execution model and launcher behavior.
