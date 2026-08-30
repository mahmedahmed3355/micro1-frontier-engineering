# Case 001 Execution Harness Contract

The harness must execute the candidate and reference independently.

For each deterministic workload size:

1. Generate deterministic input vectors.
2. Execute the reference implementation.
3. Execute the candidate implementation.
4. Synchronize before reading device results.
5. Compare every valid output element.
6. Check CUDA execution status.
7. Record execution metadata.
8. Record the candidate result in the trajectory.
9. Never use the candidate output as the expected result.
10. Never modify the reference implementation during evaluation.

The boundary matrix must include sizes below, equal to, and above the
configured CUDA block size.

A candidate passes only when all required workloads satisfy the expected
behavior.

A candidate must not pass merely because one convenient workload happens
to produce the expected numerical result.
