# Case 002 Verification Tests

The Case 002 verifier evaluates host/device transfer correctness.

The verification design must cover:

1. Small transfers.
2. Page-boundary-adjacent sizes.
3. Larger transfers.
4. Repeated transfers.
5. Complete output equality.
6. Correct CUDA error handling.
7. Device-memory lifetime.
8. Completion before the synchronous API returns.

The tests must not use the candidate output as the expected result.

The reference implementation provides the independent correctness oracle.

A passing implementation must satisfy the behavioral contract rather than
only matching one convenient transfer size.
